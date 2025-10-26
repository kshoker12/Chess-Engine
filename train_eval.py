import numpy as np
import joblib
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
import argparse
import json
import os

def load_data(prefix: str): 
    X = np.load(f"{prefix}_X.npy")
    y = np.load(f"{prefix}_y.npy").astype(np.float32)
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="dataset", help="Prefix of *_X.npy and *_y.npy")
    ap.add_argument("--out", default="sk_eval.joblib", help="Output model path")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    X, y = load_data(args.prefix)

    # Split: 80 / 10 / 10
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = args.seed)

    x_scaler = StandardScaler(with_mean=True, with_std=True)
    y_scaler = StandardScaler(with_mean=True, with_std=True)

    params_grid = {
        "mlp__regressor__hidden_layer_sizes": [ [264, 128, 64, 32, 8]],
        "mlp__regressor__alpha": [5e-5],
        "mlp__regressor__learning_rate_init": [3e-3],
    }

    base_mlp = MLPRegressor(
        activation="relu",
        solver="adam",
        random_state = args.seed,
        verbose = True,
        early_stopping = True,
        max_iter = 300,
    )

    pipe = Pipeline([
        ("xscaler", x_scaler),
        ("mlp", TransformedTargetRegressor(
            regressor=base_mlp,
            transformer=y_scaler
        ))
    ])

    # Grid search for best hyperparameters
    grid_search = GridSearchCV(pipe, params_grid, cv=10, return_train_score=True)
    grid_search.fit(X_train, y_train)

    print("Best parameters set:")
    print(grid_search.best_params_)

    print("Best score:")
    print(grid_search.best_score_)

    best_mlp = grid_search.best_estimator_

    best_mlp.fit(X_train, y_train)

    def eval_split(name, Xs, ys):
        pred = best_mlp.predict(Xs)
        mae = mean_absolute_error(ys, pred)
        r2  = r2_score(ys, pred)
        print(f"{name}: MAE={mae:.1f} cp   R2={r2:.3f}")
        return mae, r2

    print("\n=== Test ===")
    test_mae, test_r2 = eval_split("Test", X_test, y_test)

    joblib.dump(best_mlp, args.out)
    print(f"\nSaved model → {args.out}")

    # small manifest for reproducibility
    manifest = {
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "hidden": best_mlp.named_steps['mlp'].regressor_.hidden_layer_sizes,
        "alpha": best_mlp.named_steps['mlp'].regressor_.alpha,
        "lr": best_mlp.named_steps['mlp'].regressor_.learning_rate_init,
        "max_iter": best_mlp.named_steps['mlp'].regressor_.max_iter,
        "seed": args.seed,
        "test_mae_cp": float(test_mae),
        "test_r2": float(test_r2),
    }
    with open(os.path.splitext(args.out)[0] + "_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("Wrote manifest:", os.path.splitext(args.out)[0] + "_manifest.json")

if __name__ == "__main__":
    main()