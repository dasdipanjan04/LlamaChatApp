FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

WORKDIR /app

COPY src ./src/

COPY requirements.txt .

COPY config.json .

COPY entrypoint.sh /app/entrypoint.sh

RUN apt update && apt upgrade -y && \
    apt install -y software-properties-common curl wget git && \
    apt update && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH

RUN pip3 install -r requirements.txt

COPY .streamlit /app/.streamlit

EXPOSE 8000 8501

RUN useradd -m appuser && chown -R appuser /app

USER appuser

RUN chmod +x /app/entrypoint.sh

CMD ["./entrypoint.sh"]
