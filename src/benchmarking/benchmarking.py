"""Benchmarking to check"""

import asyncio
import subprocess

import httpx
import time
import statistics
import matplotlib.pyplot as plt

URL = "http://127.0.0.1:8000/query"

PROMPTS = [
    "What is the capital of France?",
    "Explain quantum mechanics.",
    "How does a black hole work?",
    "Tell me a joke.",
    "What is the future of AI?",
]


async def send_request(client, prompt):
    """
    Send prompt request to client to check latency, response

    Args:
        client: API Client responsible for doing get, post etc.
        prompt: The prompt which be passed to llama.

    Returns:
        Returns latency, response status code, response text
    """
    payload = {"queries": [prompt]}
    start_time = time.time()
    try:
        response = await client.post(URL, json=payload)
        response.raise_for_status()
        latency = time.time() - start_time
        return latency, response.status_code, response.text
    except Exception as e:
        latency = time.time() - start_time
        return latency, None, str(e)


def get_gpu_utilization():
    """
    Get GPU Utilization

    Args:

    Returns:
        Returns GPU Utilization Data
    """
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
        )
        utilization, memory_used, memory_total = map(int, output.split(","))
        return {
            "gpu_utilization": utilization,
            "memory_used": memory_used,
            "memory_total": memory_total,
        }
    except Exception as e:
        print(f"Failed to get GPU utilization: {e}")
        return {
            "gpu_utilization": 0,
            "memory_used": 0,
            "memory_total": 0,
        }


async def send_request_with_semaphore(client, prompt, semaphore):
    """
    Semaphore request

    Args:
        client: API Client
        prompt: Prompt
        semaphore: semaphore

    Returns:
        Returns GPU Utilization Data
    """
    async with semaphore:
        return await send_request(client, prompt)


async def benchmark_service(concurrent_requests, num_requests):
    async with httpx.AsyncClient(timeout=120) as client:
        semaphore = asyncio.Semaphore(concurrent_requests)

        gpu_stats_over_time = []
        start_time = time.time()

        async def log_gpu_stats(gpu_stats_over_time, start_time):
            while True:
                gpu_stats = get_gpu_utilization()
                gpu_stats["timestamp"] = time.time() - start_time
                gpu_stats_over_time.append(gpu_stats)
                await asyncio.sleep(1)

        gpu_logger_task = asyncio.create_task(
            log_gpu_stats(gpu_stats_over_time, start_time)
        )
        gpu_stats_before = get_gpu_utilization()
        tasks = [
            send_request_with_semaphore(client, PROMPTS[_ % len(PROMPTS)], semaphore)
            # send_request(client, PROMPTS[_ % len(PROMPTS)])
            for _ in range(num_requests)
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=True)
        gpu_logger_task.cancel()
        gpu_stats_after = get_gpu_utilization()
        try:
            await gpu_logger_task
        except asyncio.CancelledError:
            pass
        print(f"GPU utilisation after benchmarking {gpu_stats_after}")
        latencies = []
        errors = 0
        for result in responses:
            latency, status, _ = result
            if isinstance(result, tuple):
                if status:
                    latencies.append(latency)
                else:
                    errors += 1
            else:
                errors += 1

        total_requests = len(responses)
        successful_requests = total_requests - errors
        avg_latency = statistics.mean(latencies) if latencies else float("inf")
        max_latency = max(latencies, default=0)
        min_latency = min(latencies, default=0)
        throughput = successful_requests / sum(latencies) if latencies else 0

        print("\nBenchmark Results:")
        print(f"Total Requests: {total_requests}")
        print(f"Successful Requests: {successful_requests}")
        print(f"Failed Requests: {errors}")
        print(f"Average Latency: {avg_latency:.2f} seconds")
        print(f"Min Latency: {min_latency:.2f} seconds")
        print(f"Max Latency: {max_latency:.2f} seconds")
        print(f"Throughput: {throughput:.2f} requests/second")

        return {
            "total_requests": total_requests,
            "concurrent_requests": concurrent_requests,
            "successful_requests": successful_requests,
            "failed_requests": errors,
            "avg_latency": avg_latency,
            "min_latency": min_latency,
            "max_latency": max_latency,
            "throughput": throughput,
            "gpu_stats_before": gpu_stats_before,
            "gpu_stats_after": gpu_stats_after,
            "gpu_stats_over_time": gpu_stats_over_time,
        }


def plot_gpu_stats_over_concurrency(gpu_stats):
    """
    Plot GPU stats

    Args:
        gpu_stats: GPU Stats

    Returns:
        None
    """
    plt.figure(figsize=(12, 8))

    for concurrency, stats in gpu_stats.items():
        timestamps = [s["timestamp"] for s in stats]
        gpu_utilization = [s["gpu_utilization"] for s in stats]
        memory_used = [s["memory_used"] for s in stats]

        plt.plot(
            timestamps,
            gpu_utilization,
            label=f"Concurrency {concurrency} - GPU Utilization (%)",
        )

    plt.title("GPU Utilization Over Time for Different Concurrency Levels")
    plt.xlabel("Time (seconds)")
    plt.ylabel("GPU Utilization (%)")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 8))

    for concurrency, stats in gpu_stats.items():
        timestamps = [s["timestamp"] for s in stats]
        memory_used = [s["memory_used"] for s in stats]

        plt.plot(
            timestamps,
            memory_used,
            label=f"Concurrency {concurrency} - Memory Used (MB)",
        )

    plt.title("GPU Memory Usage Over Time for Different Concurrency Levels")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Memory Used (MB)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_avg_latency_vs_concurrency(avg_latency, concurrency):
    """
    Plot Average Latency vs Concurrency stats

    Args:
        avg_latency: Average Latency
        concurrency: Concurrency

    Returns:
        None
    """
    plt.figure()
    plt.plot(concurrency, avg_latency, marker="o")
    plt.title("Average Latency vs Concurrency")
    plt.xlabel("Concurrency Level")
    plt.ylabel("Average Latency (seconds)")
    plt.grid(True)
    plt.show()


def plot_throughput_vs_concurrency(throughput, concurrency):
    """
    Plot Throughput vs Concurrency stats

    Args:
        throughput: Throughput
        concurrency: Concurrency

    Returns:
        None
    """
    plt.figure()
    plt.plot(concurrency, throughput, marker="o")
    plt.title("Throughput vs Concurrency")
    plt.xlabel("Concurrency Level")
    plt.ylabel("Throughput (requests/second)")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    concurrency_levels = [1, 2, 3, 4]
    num_requests_per_level = 5
    results = []
    for concurrency in concurrency_levels:
        print(f"\nTesting with {concurrency} concurrent requests:")
        result = asyncio.run(
            benchmark_service(
                concurrent_requests=concurrency, num_requests=num_requests_per_level
            )
        )
        results.append((concurrency, result))
        print(f"Benchmark Results: {result}")
    concurrency = [r[0] for r in results]
    avg_latency = [r[1]["avg_latency"] for r in results]
    throughput = [r[1]["throughput"] for r in results]
    gpu_stats = {
        r[1]["concurrent_requests"]: r[1]["gpu_stats_over_time"] for r in results
    }

    plot_avg_latency_vs_concurrency(avg_latency, concurrency)
    plot_throughput_vs_concurrency(throughput, concurrency)
    plot_gpu_stats_over_concurrency(gpu_stats)
