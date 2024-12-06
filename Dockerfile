FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

WORKDIR /app

COPY src ./src/

RUN apt update && apt upgrade -y && \
    apt install -y software-properties-common curl wget git && \
    apt update && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH

COPY requirements.txt .

RUN pip3 install -r requirements.txt

EXPOSE 8000

RUN useradd -m appuser && chown -R appuser /app

USER appuser

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
