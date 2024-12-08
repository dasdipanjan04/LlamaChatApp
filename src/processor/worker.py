import asyncio
import logging
from src.processor.parallel_processor import ParallelProcessor


class Worker:
    def __init__(self, parallel_processor: ParallelProcessor, request_queue):
        self.parallel_processor = parallel_processor
        self.request_queue = request_queue

    async def queue_worker(self, semaphore):
        """
        Continuously process requests from queue with concurrency control.

        Args:
            semaphore: Semaphore for concurrency control

        Returns:
            None
        """
        while True:
            try:
                if not self.request_queue.empty():
                    func, args = await self.request_queue.get()
                    async with semaphore:
                        logging.debug(f"Processing request with args: {args}")
                        asyncio.create_task(func(*args))
                else:
                    await asyncio.sleep(0.1)
            except Exception as e:
                logging.error(f"Error in queue_worker: {e}")

    async def handle_request(self, request, response_callback):
        """
        Handle Request: Generate responses for queries.

        Args:
            request: QueryRequest instance
            response_callback: Send back the response
        """
        try:
            logging.debug(f"Handling request: {request.queries}")
            results = await self.parallel_processor.generate_parallel_responses(
                request.queries
            )
            await response_callback(results)
        except Exception as e:
            logging.error(f"Error handling request: {e}")
            await response_callback({"error": str(e)})
