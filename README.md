docker build -t fastapi-llama-service .
docker run -it --rm --gpus all -p 8000:8000 fastapi-llama-service