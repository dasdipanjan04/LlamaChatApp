import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from src.pipeline.pipeline import model_manager

global_thread_index = 0
thread_index_lock = asyncio.Lock()


async def process_prompt(prompt, idx=None):
    global global_thread_index

    async with thread_index_lock:
        thread_index = global_thread_index
        global_thread_index += 1

    start_time = time.time()
    print(f"Thread {thread_index} started at {start_time:.2f}")
    print(f"Thread {thread_index} Response: ", end="", flush=True)

    response = ""
    token_count = 0

    async for token in model_manager.generate_response(prompt):
        print(
            f"\nThread {thread_index}: Word{token_count}_{thread_index} -> {token}",
            end="",
            flush=True,
        )
        response += token
        token_count += 1

    end_time = time.time()
    print(
        f"\nThread {thread_index} completed at {end_time:.2f} (Duration: {end_time - start_time:.2f}s)"
    )
    return response


async def generate_parallel_responses(prompts):
    results = {}
    with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(executor, asyncio.run, process_prompt(prompt, idx))
            for idx, prompt in enumerate(prompts)
        ]
        responses = await asyncio.gather(*tasks)

    for idx, response in enumerate(responses):
        results[idx] = response

    return results
