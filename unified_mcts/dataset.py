"""Memmap datasets for value / policy training and FT mixes."""
from __future__ import annotations

import json
import os
import pickle
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import chess
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from board_utils import (
    encode_board_stm_white,
    get_uci4_token,
    mirror_uci_string,
    pad_history,
)


def _open_memmap(path: str, dtype, shape, mode: str = "r"):
    return np.memmap(path, dtype=dtype, mode=mode, shape=tuple(shape))


class ValueMemmapDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train", val_ratio: float = 0.1, seed: int = 42):
        meta_path = os.path.join(data_dir, "value_meta.json")
        with open(meta_path) as f:
            self.meta = json.load(f)
        n = int(self.meta["N"])
        shape_x = (n, 64)
        shape_r = (n, int(self.meta.get("K", 8)))
        self.X = _open_memmap(os.path.join(data_dir, "value_X.u8.memmap"), np.uint8, shape_x)
        self.R = _open_memmap(os.path.join(data_dir, "value_R.u8.memmap"), np.uint8, shape_r)
        self.Y = _open_memmap(os.path.join(data_dir, "value_Y.f16.memmap"), np.float16, (n,))

        rng = np.random.RandomState(seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_val = int(n * val_ratio)
        if split == "train":
            self.indices = idx[n_val:]
        else:
            self.indices = idx[:n_val]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        j = int(self.indices[i])
        board = torch.from_numpy(np.array(self.X[j], dtype=np.int64))
        rights = torch.from_numpy(np.array(self.R[j], dtype=np.int64))
        y = torch.tensor(float(self.Y[j]), dtype=torch.float32)
        return board, rights, y


class PolicyMemmapDataset(Dataset):
    """Single corpus memmap. policy_yv is optional (zeros if missing)."""

    def __init__(self, data_dir: str, start: int = 0, end: Optional[int] = None):
        meta_path = os.path.join(data_dir, "policy_meta.json")
        with open(meta_path) as f:
            self.meta = json.load(f)
        n = int(self.meta["N"])
        self.block_size = int(self.meta.get("block_size", 128))
        self.K = int(self.meta.get("K", 8))
        end = n if end is None else end
        self.start = start
        self.end = end
        self.hist = _open_memmap(
            os.path.join(data_dir, "policy_hist.u16.memmap"),
            np.uint16,
            (n, self.block_size),
        )
        self.board = _open_memmap(
            os.path.join(data_dir, "policy_board.u8.memmap"), np.uint8, (n, 64)
        )
        self.rights = _open_memmap(
            os.path.join(data_dir, "policy_R.u8.memmap"), np.uint8, (n, self.K)
        )
        self.y = _open_memmap(
            os.path.join(data_dir, "policy_y.u16.memmap"), np.uint16, (n,)
        )
        yv_path = os.path.join(data_dir, "policy_yv.f16.memmap")
        if os.path.exists(yv_path):
            self.yv = _open_memmap(yv_path, np.float16, (n,))
            self.has_yv = True
        else:
            self.yv = None
            self.has_yv = False

    def __len__(self) -> int:
        return self.end - self.start

    def __getitem__(self, i: int):
        j = self.start + int(i)
        hist = torch.from_numpy(np.array(self.hist[j], dtype=np.int64))
        board = torch.from_numpy(np.array(self.board[j], dtype=np.int64))
        rights = torch.from_numpy(np.array(self.rights[j], dtype=np.int64))
        y = torch.tensor(int(self.y[j]), dtype=torch.long)
        if self.has_yv:
            yv = torch.tensor(float(self.yv[j]), dtype=torch.float32)
        else:
            yv = torch.tensor(0.0, dtype=torch.float32)
        return hist, board, rights, y, yv


class SourceMixSampler(Sampler[int]):
    """Sample global memmap indices with per-source mix weights (no O(N) weight tensor)."""

    def __init__(
        self,
        ranges: Sequence[Tuple[int, int]],
        weights: Sequence[float],
        num_samples: int,
        seed: int = 42,
    ):
        assert len(ranges) == len(weights) and len(ranges) > 0
        self.ranges = [(int(a), int(b)) for a, b in ranges]
        w = np.asarray(weights, dtype=np.float64)
        w = np.clip(w, 0.0, None)
        if w.sum() <= 0:
            w = np.ones_like(w)
        self.probs = w / w.sum()
        self.num_samples = int(num_samples)
        self.rng = np.random.RandomState(seed)

    def __iter__(self) -> Iterator[int]:
        for _ in range(self.num_samples):
            s = int(self.rng.choice(len(self.ranges), p=self.probs))
            start, end = self.ranges[s]
            yield int(self.rng.randint(start, end))

    def __len__(self) -> int:
        return self.num_samples


class MixedPolicyDataset(Dataset):
    """
    Policy corpus with 80/10/10 (or custom) source mix at sample time.
    meta.json sources: [{name, start, end}, ...]
    """

    def __init__(self, data_dir: str, mix: Optional[Dict[str, float]] = None):
        mix = mix or {"full": 0.80, "elite": 0.10, "weak": 0.10}
        meta_path = os.path.join(data_dir, "policy_meta.json")
        with open(meta_path) as f:
            self.meta = json.load(f)
        if self.meta.get("format") == "pt_replay":
            raise ValueError("Use PtReplayPolicyDataset for format=pt_replay")
        self.base = PolicyMemmapDataset(data_dir)
        sources = self.meta.get("sources")
        if not sources:
            sources = [{"name": "full", "start": 0, "end": int(self.meta["N"])}]

        self.ranges: List[Tuple[int, int]] = []
        self.source_weights: List[float] = []
        self.source_names: List[str] = []
        for src in sources:
            name = src["name"]
            w = float(mix.get(name, 0.0))
            start, end = int(src["start"]), int(src["end"])
            if w <= 0 or end <= start:
                continue
            self.ranges.append((start, end))
            self.source_weights.append(w)
            self.source_names.append(name)

        if not self.ranges:
            n = int(self.meta["N"])
            self.ranges = [(0, n)]
            self.source_weights = [1.0]
            self.source_names = ["all"]

        self._len = int(self.meta["N"])

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, j: int):
        j = int(j)
        hist = torch.from_numpy(np.array(self.base.hist[j], dtype=np.int64))
        board = torch.from_numpy(np.array(self.base.board[j], dtype=np.int64))
        rights = torch.from_numpy(np.array(self.base.rights[j], dtype=np.int64))
        y = torch.tensor(int(self.base.y[j]), dtype=torch.long)
        if self.base.has_yv:
            yv = torch.tensor(float(self.base.yv[j]), dtype=torch.float32)
        else:
            yv = torch.tensor(0.0, dtype=torch.float32)
        return hist, board, rights, y, yv

    def make_sampler(self, num_samples: Optional[int] = None, seed: int = 42) -> SourceMixSampler:
        n = num_samples or self._len
        return SourceMixSampler(self.ranges, self.source_weights, n, seed=seed)


def _first_move_legal(uci: str) -> bool:
    b = chess.Board()
    try:
        b.push_uci(uci)
        return True
    except ValueError:
        return False


def _push_uci4(board: chess.Board, uci4: str) -> bool:
    """Push a 4-char UCI token; vocab has no promo piece, so fall back to queen."""
    try:
        board.push_uci(uci4)
        return True
    except ValueError:
        pass
    try:
        board.push_uci(uci4 + "q")
        return True
    except ValueError:
        return False


def replay_policy_row(
    x_stream: np.ndarray,
    y_stream: np.ndarray,
    game_start: int,
    end: int,
    itos: dict,
    pad_id: int,
    block_size: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    """
    Replay framed tokens x[game_start:end+1], return (hist, board, rights, y_id).
    If the token stream is a Black-win mirror, un-mirror for physical push.
    """
    if end < game_start:
        return None
    framed = [int(x_stream[k]) for k in range(game_start, end + 1)]
    if not framed:
        return None
    u0 = itos.get(framed[0])
    if not u0 or len(u0) < 4:
        return None
    mirrored = not _first_move_legal(get_uci4_token(u0))

    board = chess.Board()
    for tid in framed:
        u = itos.get(tid)
        if not u or len(u) < 4:
            return None
        phys = mirror_uci_string(get_uci4_token(u)) if mirrored else get_uci4_token(u)
        if not _push_uci4(board, phys):
            return None

    y_id = int(y_stream[end])
    if y_id < 0 or y_id == pad_id:
        return None
    pieces, rights, _ = encode_board_stm_white(board)
    hist = pad_history(framed, block_size, pad_id)
    return hist, pieces, rights, y_id


class PtReplayPolicyDataset(Dataset):
    """
    80/10/10 mix over full/elite/weak .pt streams with on-the-fly board replay.
    Requires policy_meta.json with format=pt_replay from build_policy_from_pt.py.
    """

    def __init__(self, data_dir: str, mix: Optional[Dict[str, float]] = None, seed: int = 42):
        mix = mix or {"full": 0.80, "elite": 0.10, "weak": 0.10}
        meta_path = os.path.join(data_dir, "policy_meta.json")
        with open(meta_path) as f:
            self.meta = json.load(f)
        if self.meta.get("format") != "pt_replay":
            raise ValueError("PtReplayPolicyDataset expects format=pt_replay")

        with open(self.meta["vocab"], "rb") as f:
            stoi = pickle.load(f)
        self.itos = {int(i): s for s, i in stoi.items()}
        self.pad_id = int(self.meta["pad_id"])
        self.block_size = int(self.meta.get("block_size", 128))

        self.sources = []
        weights = []
        for src in self.meta["sources"]:
            name = src["name"]
            w = float(mix.get(name, 0.0))
            if w <= 0 or int(src["n_rows"]) <= 0:
                continue
            if src.get("x_memmap"):
                x = _open_memmap(
                    os.path.join(data_dir, src["x_memmap"]),
                    np.int16,
                    (int(src["stream_len"]),),
                )
                y = _open_memmap(
                    os.path.join(data_dir, src["y_memmap"]),
                    np.int16,
                    (int(src["stream_len"]),),
                )
            else:
                x = np.asarray(
                    torch.load(src["x_pt"], map_location="cpu", weights_only=True),
                    dtype=np.int16,
                )
                y = np.asarray(
                    torch.load(src["y_pt"], map_location="cpu", weights_only=True),
                    dtype=np.int16,
                )
            n_rows = int(src["n_rows"])
            ends = _open_memmap(
                os.path.join(data_dir, src["end_memmap"]), np.uint32, (n_rows,)
            )
            starts = _open_memmap(
                os.path.join(data_dir, src["start_memmap"]), np.uint32, (n_rows,)
            )
            self.sources.append(
                {"name": name, "x": x, "y": y, "ends": ends, "starts": starts, "n": n_rows}
            )
            weights.append(w)

        if not self.sources:
            raise ValueError("No usable sources in policy_meta.json")
        w = np.asarray(weights, dtype=np.float64)
        self.source_probs = w / w.sum()
        self.rng = np.random.RandomState(seed)
        self._len = int(self.meta["N"])

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, _i: int):
        s = int(self.rng.choice(len(self.sources), p=self.source_probs))
        src = self.sources[s]
        for _ in range(8):
            k = int(self.rng.randint(0, src["n"]))
            row = replay_policy_row(
                src["x"],
                src["y"],
                int(src["starts"][k]),
                int(src["ends"][k]),
                self.itos,
                self.pad_id,
                self.block_size,
            )
            if row is not None:
                hist, board, rights, y_id = row
                return (
                    torch.from_numpy(np.asarray(hist, dtype=np.int64)),
                    torch.from_numpy(np.asarray(board, dtype=np.int64)),
                    torch.from_numpy(np.asarray(rights, dtype=np.int64)),
                    torch.tensor(y_id, dtype=torch.long),
                    torch.tensor(0.0, dtype=torch.float32),
                )
        hist = pad_history([], self.block_size, self.pad_id)
        board = np.zeros(64, dtype=np.uint8)
        rights = np.zeros(8, dtype=np.uint8)
        return (
            torch.from_numpy(np.asarray(hist, dtype=np.int64)),
            torch.from_numpy(board.astype(np.int64)),
            torch.from_numpy(rights.astype(np.int64)),
            torch.tensor(self.pad_id, dtype=torch.long),
            torch.tensor(0.0, dtype=torch.float32),
        )

    def make_sampler(self, num_samples: Optional[int] = None, seed: int = 42) -> Sampler:
        n = num_samples or self._len
        return SourceMixSampler([(0, n)], [1.0], n, seed=seed)


class MixedValueFTDataset(Dataset):
    """
    Mix self-play value FT data with base Lichess value data.
    Returns (board, rights, y_v). Sample fraction `ft_frac` from FT, rest from base.
    """

    def __init__(
        self,
        ft_dir: str,
        base_dir: Optional[str] = None,
        ft_frac: float = 0.8,
        seed: int = 42,
    ):
        self.ft = ValueMemmapDataset(ft_dir, split="train", val_ratio=0.0, seed=seed)
        self.base = ValueMemmapDataset(base_dir, split="train", val_ratio=0.0, seed=seed) if base_dir else None
        self.ft_frac = float(ft_frac)
        self.rng = np.random.RandomState(seed)
        self._len = max(len(self.ft), 1)
        if self.base is not None:
            self._len = max(self._len, len(self.base))

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, i: int):
        use_ft = self.base is None or (self.rng.rand() < self.ft_frac)
        ds = self.ft if use_ft else self.base
        j = int(self.rng.randint(0, len(ds)))
        return ds[j]


class MixedPolicyFTDataset(Dataset):
    """
    Mix self-play policy FT with base policy memmap.
    Returns (hist, board, rights, y_pi, y_v).
    """

    def __init__(
        self,
        ft_dir: str,
        base_dir: Optional[str] = None,
        ft_frac: float = 0.8,
        seed: int = 42,
    ):
        self.ft = PolicyMemmapDataset(ft_dir)
        self.base = PolicyMemmapDataset(base_dir) if base_dir else None
        self.ft_frac = float(ft_frac)
        self.rng = np.random.RandomState(seed)
        self._len = max(len(self.ft), 1)
        if self.base is not None:
            self._len = max(self._len, len(self.base))

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, i: int):
        use_ft = self.base is None or (self.rng.rand() < self.ft_frac)
        ds = self.ft if use_ft else self.base
        j = int(self.rng.randint(0, len(ds)))
        return ds[j]


class MixedPolicyPtReplayFTDataset(Dataset):
    """
    Mix self-play policy memmap FT rows with a pt_replay base corpus.
    Returns (hist, board, rights, y_pi, y_v) — same as PolicyMemmapDataset.
    """

    def __init__(
        self,
        ft_dir: str,
        base_dir: Optional[str] = None,
        ft_frac: float = 0.7,
        seed: int = 42,
        split: str = "train",
        val_ratio: float = 0.05,
    ):
        self.ft = PolicyMemmapDataset(ft_dir)
        n = len(self.ft)
        rng = np.random.RandomState(seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_val = int(n * val_ratio)
        self.ft_indices = idx[:n_val] if split == "val" else idx[n_val:]
        if len(self.ft_indices) == 0:
            self.ft_indices = idx

        self.base = None
        if base_dir:
            meta_path = os.path.join(base_dir, "policy_meta.json")
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("format") == "pt_replay":
                self.base = PtReplayPolicyDataset(base_dir, seed=seed + 1)
            else:
                self.base = PolicyMemmapDataset(base_dir)
        self.ft_frac = float(ft_frac)
        self.rng = np.random.RandomState(seed + (0 if split == "train" else 17))
        self._len = max(len(self.ft_indices), 1)
        if self.base is not None:
            self._len = max(self._len, len(self.base))

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, i: int):
        use_ft = self.base is None or (self.rng.rand() < self.ft_frac)
        if use_ft:
            j = int(self.ft_indices[int(self.rng.randint(0, len(self.ft_indices)))])
            return self.ft[j]
        j = int(self.rng.randint(0, len(self.base)))
        return self.base[j]


def write_memmap(path: str, arr: np.ndarray):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    mm = np.memmap(path, dtype=arr.dtype, mode="w+", shape=arr.shape)
    mm[:] = arr
    mm.flush()
    del mm


class StreamingPolicyWriter:
    """Append policy rows to raw memmap files without holding the full corpus in RAM."""

    def __init__(self, out_dir: str, block_size: int, rights_dim: int, flush_every: int = 50_000):
        self.out_dir = out_dir
        self.block_size = block_size
        self.rights_dim = rights_dim
        self.flush_every = flush_every
        os.makedirs(out_dir, exist_ok=True)
        self.paths = {
            "hist": os.path.join(out_dir, "policy_hist.u16.memmap"),
            "board": os.path.join(out_dir, "policy_board.u8.memmap"),
            "rights": os.path.join(out_dir, "policy_R.u8.memmap"),
            "y": os.path.join(out_dir, "policy_y.u16.memmap"),
            "yv": os.path.join(out_dir, "policy_yv.f16.memmap"),
        }
        for p in self.paths.values():
            open(p, "wb").close()
        self.n = 0
        self._hist: List[np.ndarray] = []
        self._board: List[np.ndarray] = []
        self._rights: List[np.ndarray] = []
        self._y: List[np.ndarray] = []
        self._yv: List[np.ndarray] = []

    def add(self, hist, board, rights, y, yv) -> None:
        self._hist.append(np.asarray(hist, dtype=np.uint16))
        self._board.append(np.asarray(board, dtype=np.uint8))
        self._rights.append(np.asarray(rights, dtype=np.uint8))
        self._y.append(np.asarray(y, dtype=np.uint16))
        self._yv.append(np.asarray(yv, dtype=np.float16))
        if len(self._y) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._y:
            return
        hist = np.stack(self._hist).astype(np.uint16, copy=False)
        board = np.stack(self._board).astype(np.uint8, copy=False)
        rights = np.stack(self._rights).astype(np.uint8, copy=False)
        y = np.asarray(self._y, dtype=np.uint16)
        yv = np.asarray(self._yv, dtype=np.float16)
        with open(self.paths["hist"], "ab") as f:
            f.write(hist.tobytes())
        with open(self.paths["board"], "ab") as f:
            f.write(board.tobytes())
        with open(self.paths["rights"], "ab") as f:
            f.write(rights.tobytes())
        with open(self.paths["y"], "ab") as f:
            f.write(y.tobytes())
        with open(self.paths["yv"], "ab") as f:
            f.write(yv.tobytes())
        self.n += len(y)
        self._hist.clear()
        self._board.clear()
        self._rights.clear()
        self._y.clear()
        self._yv.clear()

    def close(self) -> int:
        self.flush()
        return self.n
