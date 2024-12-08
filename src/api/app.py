from fastapi import FastAPI
from slowapi import Limiter
from slowapi.util import get_remote_address
from contextlib import asynccontextmanager
from src.api.endpoints import router as api_router
from src.api.error_handlers import add_exception_handlers
from src.core.model_manager import ModelManager, AbuseDetector
from src.processor.parallel_processor import ParallelProcessor
from src.processor.worker import Worker
from src.processor.thread_manager import ThreadManager
from src.log_config import configure_logging
from src.config import DEBUG_MODE, MAX_CONCURRENT_REQUESTS, RATE_LIMIT
import asyncio
import logging

configure_logging(DEBUG_MODE)

request_queue = asyncio.Queue()
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
model_manager = ModelManager(AbuseDetector())
parallel_processor = ParallelProcessor(model_manager, ThreadManager())
worker = Worker(parallel_processor, request_queue)
limiter = Limiter(key_func=get_remote_address)


def create_app() -> FastAPI:
    """
    Creates and configures a FastAPI application.

    Args:

    Returns:
        FastAPI: Application instance
    """
    app = FastAPI(lifespan=lifespan)
    app.state.limiter = limiter
    app.state.rate_limit = RATE_LIMIT
    app.state.worker = worker

    app.include_router(api_router)
    add_exception_handlers(app)

    return app


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan: Setup and cleanup logic.

    Args:
        app (FastAPI): App Instance
    Returns:

    """
    await model_manager.load_model_async()
    logging.info("Model loaded successfully")

    worker_task = asyncio.create_task(worker.queue_worker(semaphore))

    try:
        yield
    finally:
        logging.info("Shutting Down Application")
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            logging.error("Worker task cancelled")
