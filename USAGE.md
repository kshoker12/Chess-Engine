```
export DOCKER_BUILDKIT=0                 
                    
docker build --platform linux/arm64 -t chess-ai-backend:latest .

aws ecr get-login-password --region us-east-2 | \
  docker login --username AWS --password-stdin 950538587033.dkr.ecr.us-east-2.amazonaws.com

docker tag chess-ai-backend:latest \
  950538587033.dkr.ecr.us-east-2.amazonaws.com/chess-ai-backend:latest

docker push 950538587033.dkr.ecr.us-east-2.amazonaws.com/chess-ai-backend:latest
```