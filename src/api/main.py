import asyncio
from asyncio import Queue, create_task

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from contextlib import asynccontextmanager
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from src.pipeline import model_manager, generate_parallel_responses
from src.log_config import configure_logging
from src.config import DEBUG_MODE, MAX_CONCURRENT_REQUESTS, RATE_LIMIT
from src.metrics import metrics_endpoint
import logging


request_queue = Queue()
configure_logging(DEBUG_MODE)


class QueryRequest(BaseModel):
    """ " Query request and validation"""

    queries: list[str] = Field(..., description="List of queries")

    @field_validator("queries", mode="before")
    def validate_queries(cls, queries):
        """
        Validate Queries Field Validator

        Args:
            cls: QueryRequest class
            queries: list of queries

        Returns:
            None
        """
        if not queries:
            raise ValueError("Queries cannot be empty.")
        return queries


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Fast API Lifespan

    Args:
        app: FastAPI

    Returns:
        None
    """
    await model_manager.load_model_async()
    logging.info("Model loaded successfully")

    worker_task = create_task(queue_worker())

    yield

    logging.info("Shutting Down Application")
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        logging.error("Worker task cancelled")


app = FastAPI(lifespan=lifespan)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.state.rate_limit = RATE_LIMIT


async def queue_worker():
    """
    Continuously process requests from queue with concurrency control.

    Args:

    Returns:
        None
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    while True:
        if not request_queue.empty():
            func, args = await request_queue.get()
            async with semaphore:
                logging.debug(f"Processing request with args: {args}")
                asyncio.create_task(func(*args))
        else:
            await asyncio.sleep(0.1)


async def handle_request(request, response_callback):
    """
    Handle Request

    Args:
        request: Request
        response_callback: Response Callback
    Returns:
        None
    """
    logging.debug(f"Handling request: {request.queries}")
    results = await generate_parallel_responses(request.queries)
    await response_callback(results)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """
    Rate limiter

    Args:
        request: Request
        exc: Rate limit exceeded exception handler
    Returns:
        Jsonresponse with status code and detailed error message
    """
    return JSONResponse(
        status_code=429,
        content={
            "detail": "The allowed rate limit has been exceeded. PLease try again later"
        },
    )


@app.get("/health")
async def health_check():
    """
    Health Check GET Method

    Args:

    Returns:
        Json
    """
    return {"status": "healthy"}


@app.post("/query")
@limiter.limit(RATE_LIMIT)
async def process_query(request: Request, query: QueryRequest):
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

    await request_queue.put((handle_request, (query, response_callback)))
    return await response


@app.get("/metrics")
async def metrics():
    return await metrics_endpoint()


app.add_api_route("/metrics", metrics_endpoint)
