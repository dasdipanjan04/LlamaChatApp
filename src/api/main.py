import uuid
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from threading import Thread
from typing import AsyncGenerator
from src.pipeline.pipeline import model_manager
from src.pipeline.batch_processor import BatchProcessor

batch_processor = BatchProcessor(model_manager, batch_timeout=0.05)

batch_processor_thread = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global batch_processor_thread

    await model_manager.load_model_async()
    print("Model loaded successfully")

    batch_processor_thread = Thread(target=batch_processor.start)
    batch_processor_thread.start()

    yield

    print("Shutting Down Application")

    batch_processor.stop()
    batch_processor_thread.join()

app = FastAPI(lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def process_query(body: QueryRequest) -> StreamingResponse:
    request_id = str(uuid.uuid4())
    response_queue = await batch_processor.enqueue_request(request_id, body.query)

    async def stream_response() -> AsyncGenerator[str, None]:
        while True:
            token = await response_queue.get()
            if token is None:  # End of response
                break
            yield token

    return StreamingResponse(
        stream_response(),
        media_type="text/plain"
    )
