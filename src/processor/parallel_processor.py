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

    async def process_prompt(self, prompt: str):
        """
        Process prompt to generate response asyncly

        Args:
            prompt: The prompt which be passed to the llama response generator

        Returns:
            Returns the response
        """
        thread_index = await self.thread_manager.get_next_thread_index()

        start_time = time.time()
        print(f"Thread {thread_index} started at {start_time:.2f}")
        print(f"Thread {thread_index} Response: ", end="", flush=True)

        response = ""
        token_count = 0
        try:
            async for token in self.model_manager.generate_response(prompt):
                print(
                    f"\nThread {thread_index}: Word{token_count}_{thread_index} -> {token}",
                    end="",
                    flush=True,
                )
                response += token
                token_count += 1
        except Exception as e:
            logging.error(f"Thread {thread_index} encountered an error: {e}")
            response = f"Error generating response: {e}"
        end_time = time.time()
        print(
            f"\nThread {thread_index} completed at {end_time:.2f} (Duration: {end_time - start_time:.2f}s)"
        )
        return response

    async def generate_parallel_responses(self, prompts: list[str]):
        """
        Generate parallel responses for multiple prompts.

        Args:
            prompts (list[str]): List of prompts.

        Returns:
            dict: A dictionary of thread indices to responses.
        """
        results = {}
        with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(executor, asyncio.run, self.process_prompt(prompt))
                for prompt in prompts
            ]
            responses = await asyncio.gather(*tasks)

        for idx, response in enumerate(responses):
            results[idx] = response

        return results
