import asyncio
from asyncio import Queue, create_task
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from src.pipeline import ModelManager, generate_parallel_responses
from src.log_config import configure_logging
import logging
model_manager = ModelManager()
debug_mode = True
configure_logging(debug_mode)
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter

request_queue = Queue()
MAX_CONCURRENT_REQUESTS = 20


class QueryRequest(BaseModel):
    queries: list[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    await model_manager.load_model_async()
    logging.info("Model loaded successfully")

    worker_task = create_task(queue_worker())

    yield

    logging.info("Shutting Down Application")
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        logging.ERROR("Worker task cancelled")


app = FastAPI(lifespan=lifespan)


async def queue_worker():
    while True:
        if not request_queue.empty():
            func, args = await request_queue.get()
            logging.debug(f"Processing request with args: {args}")
            asyncio.create_task(func(*args))
        else:
            await asyncio.sleep(0.1)


async def handle_request(request, response_callback):
    logging.debug(f"Handling request: {request.queries}")
    results = await generate_parallel_responses(request.queries)
    await response_callback(results)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={
            "detail": "The allowed rate limit has been exceeded. PLease try again later"
        },
    )


@app.post("/query")
@limiter.limit("10000/minute")
async def process_query(request: Request, query: QueryRequest):
    response = asyncio.Future()

    async def response_callback(results):
        response.set_result(results)

    await request_queue.put((handle_request, (query, response_callback)))
    return await response
