from fastapi import APIRouter, Request, Depends
from src.api.validators import QueryRequest
from src.metrics import metrics_endpoint
import asyncio

router = APIRouter()


def get_worker(request: Request):
    return request.app.state.worker


@router.get("/health")
async def health_check():
    """
    Health Check GET Method

    Args:

    Returns:
        Json
    """
    return {"status": "healthy"}


@router.post("/query")
async def process_query(
    request: Request, query: QueryRequest, worker=Depends(get_worker)
):
    """
    Process Query

    Args:
        request: Request
        query: Query request, list of queries
    Returns:
        response
    """
    response = asyncio.Future()

    async def response_callback(results):
        response.set_result(results)

    await worker.request_queue.put((worker.handle_request, (query, response_callback)))
    return await response


@router.get("/metrics")
async def metrics():
    """
    Metrics

    Args:

    Returns:

    """
    return await metrics_endpoint()
