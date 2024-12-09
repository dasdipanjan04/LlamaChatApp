import asyncio
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from src.core.model_manager import ModelManager
from src.processor.thread_manager import ThreadManager


class ParallelProcessor:
    def __init__(self, model_manager: ModelManager, thread_manager: ThreadManager):
        self.model_manager = model_manager
        self.thread_manager = thread_manager

    async def stream_to_endpoint(self, message, response_callback):
        """
        Stream data to the endpoint using the provided callback.

        Args:
            message: The message to stream.
            response_callback: Callback to push the message to the endpoint.
        Return:

        """
        await response_callback(message)

    async def process_prompt(self, prompt: str, response_callback):
        """
        Process prompt to generate response asyncly and stream updates.

        Args:
            prompt: The prompt to process.
            response_callback: Callback to stream updates to the endpoint.

        Returns:
            The final response.
        """
        thread_index = await self.thread_manager.get_next_thread_index()

        response = ""
        token_count = 0
        try:
            async for token in self.model_manager.generate_response(prompt):
                await self.stream_to_endpoint(
                    token,
                    response_callback,
                )
                response += token
                token_count += 1
        except Exception as e:
            logging.error(f"Thread {thread_index} encountered an error: {e}")
            response = f"Error generating response: {e}"
            await self.stream_to_endpoint(response, response_callback)
        return response

    async def generate_parallel_responses(self, prompts: list[str], response_callback):
        """
        Generate parallel responses for multiple prompts.

        Args:
            prompts (list[str]): List of prompts.
            response_callback: Callback to stream updates to the endpoint.

        Returns:
            dict: A dictionary of thread indices to responses.
        """
        with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(
                    executor,
                    asyncio.run,
                    self.process_prompt(prompt, response_callback),
                )
                for prompt in prompts
            ]

            await asyncio.gather(*tasks)

        await response_callback("")
