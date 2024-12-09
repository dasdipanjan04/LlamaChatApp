from fastapi import APIRouter, Request, Depends
from fastapi.responses import StreamingResponse
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

    async def stream_results():
        """
        Stream results to the client.

        Yields:
            str: Incremental updates and final results.
        """
        queue = asyncio.Queue()

        async def response_callback(results):
            """
            Response callback to send results incrementally or as the final response.

            Args:
                results (dict): The result to send.
            """
            if isinstance(results, dict) and "final" in results:
                await queue.put(f"{results['final']} \n\n")
                await queue.put(None)
            else:
                await queue.put(f"{results} \n\n")

        await worker.request_queue.put(
            (worker.handle_request, (query, response_callback))
        )

        while True:
            update = await queue.get()
            if update is None:
                break
            yield update

    return StreamingResponse(stream_results(), media_type="text/event-stream")


@router.get("/metrics")
async def metrics():
    """
    Metrics

    Args:

    Returns:

    """
    return await metrics_endpoint()
