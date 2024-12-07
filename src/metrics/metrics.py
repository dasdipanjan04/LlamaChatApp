import logging

from prometheus_client import (
    generate_latest,
    CONTENT_TYPE_LATEST,
    Counter,
    Summary,
    Gauge,
)
from starlette.responses import Response
import subprocess

REQUEST_COUNT = Counter("api_request_count", "Total number of API requests")
LATENCY_SUMMARY = Summary(
    "api_request_latency_seconds", "API request latency in seconds"
)
GPU_UTILIZATION = Gauge("gpu_utilization", "GPU utilization in percentage")


def collect_gpu_metrics():
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
        )
        utilization = int(output.strip())
        GPU_UTILIZATION.set(utilization)
    except Exception as e:
        logging.error(e)
        GPU_UTILIZATION.set(0)


async def metrics_endpoint():
    collect_gpu_metrics()

    metrics = generate_latest()
    return Response(content=metrics, media_type=CONTENT_TYPE_LATEST)
