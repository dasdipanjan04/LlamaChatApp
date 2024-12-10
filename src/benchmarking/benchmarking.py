"""Benchmarking to check"""

import asyncio
import subprocess
import argparse
import httpx
import time
import statistics
import matplotlib.pyplot as plt
import psutil
import logging
import csv
import json


async def send_request(url, client, prompt):
    """
    Send prompt request to client to check latency, response

    Args:
        client: API Client responsible for doing get, post etc.
        prompt: The prompt which be passed to llama.

    Returns:
        Returns latency, response status code, response text
    """
    payload = {"queries": prompt}
    start_time = time.time()
    try:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        latency = time.time() - start_time
        return latency, response.status_code, response.text
    except Exception as e:
        return None, None, str(e)


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
        return {
            "gpu_utilization": 0,
            "memory_used": 0,
            "memory_total": 0,
        }


def get_cpu_utilization():
    """
    Get CPU Utilization

    Args:

    Returns:
        Returns CPU Utilization Data
    """
    try:
        cpu_percent = psutil.cpu_percent(interval=None)
        memory_info = psutil.virtual_memory()
        return {
            "cpu_utilization": cpu_percent,
            "memory_used": memory_info.used // (1024 * 1024),
            "memory_total": memory_info.total // (1024 * 1024),
        }
    except Exception:
        return {"cpu_utilization": None, "memory_used": None, "memory_total": None}


async def log_resource_stats(gpu_stats_over_time, cpu_stats_over_time, start_time):
    """
    Log GPU and CPU Utilization

    Args:

    Returns:
        None
    """
    while True:
        gpu_stat = get_gpu_utilization()
        cpu_stat = get_cpu_utilization()
        timestamp = time.time() - start_time

        gpu_stat["timestamp"] = timestamp
        cpu_stat["timestamp"] = timestamp

        gpu_stats_over_time.append(gpu_stat)
        cpu_stats_over_time.append(cpu_stat)

        await asyncio.sleep(1)


async def send_request_with_semaphore(url, client, prompt, semaphore):
    """
    Semaphore request

    Args:
        url: url
        client: API Client
        prompt: Prompt
        semaphore: semaphore

    Returns:
        Returns GPU Utilization Data
    """
    async with semaphore:
        return await send_request(url, client, prompt)


async def benchmark_service(url, concurrent_requests, num_requests):
    """
    Benchmark Service

    Args:
        concurrent_requests:
        num_requests:

    Returns:
        Returns Metrics
    """
    async with httpx.AsyncClient(timeout=120) as client:
        semaphore = asyncio.Semaphore(concurrent_requests)

        gpu_stats_over_time = []
        cpu_stats_over_time = []
        start_time = time.time()

        resource_logger_task = asyncio.create_task(
            log_resource_stats(gpu_stats_over_time, cpu_stats_over_time, start_time)
        )
        gpu_stats_before = get_gpu_utilization()
        cpu_stats_before = get_cpu_utilization()
        tasks = [
            send_request_with_semaphore(url, client, prompts, semaphore)
            for _ in range(num_requests)
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=True)
        resource_logger_task.cancel()
        gpu_stats_after = get_gpu_utilization()
        cpu_stats_after = get_cpu_utilization()
        try:
            await resource_logger_task
        except asyncio.CancelledError:
            pass
        logging.debug(f"GPU utilisation after benchmarking {gpu_stats_after}")
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

        logging.debug("\nBenchmark Results:")
        logging.debug(f"Total Requests: {total_requests}")
        logging.debug(f"Successful Requests: {successful_requests}")
        logging.debug(f"Failed Requests: {errors}")
        logging.debug(f"Average Latency: {avg_latency:.2f} seconds")
        logging.debug(f"Min Latency: {min_latency:.2f} seconds")
        logging.debug(f"Max Latency: {max_latency:.2f} seconds")
        logging.debug(f"Throughput: {throughput:.2f} requests/second")

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
            "cpu_stats_before": cpu_stats_before,
            "cpu_stats_after": cpu_stats_after,
            "cpu_stats_over_time": cpu_stats_over_time,
        }


def plot_resource_stats_over_concurrency(resource_stats, resource: str):
    """
    Plot Resource stats

    Args:
        resource_stats: Resource Stats
        resource: Resource name
    Returns:
        None
    """
    metrics = [
        (
            f"{resource}_utilization",
            f"{resource.upper()} Utilization (%)",
            "Utilization",
        ),
        ("memory_used", "Memory Used (MB)", "Memory Usage"),
    ]

    for metric_key, y_label, title_suffix in metrics:
        plt.figure(figsize=(12, 8))

        for concurrency, stats in resource_stats.items():
            timestamps = [s["timestamp"] for s in stats]
            metric_values = [s[metric_key] for s in stats]

            plt.plot(
                timestamps,
                metric_values,
                label=f"Concurrency {concurrency} - {y_label}",
            )

        plt.title(
            f"{resource.upper()} {title_suffix} Over Time for Different Concurrency Levels"
        )
        plt.xlabel("Time (seconds)")
        plt.ylabel(y_label)
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{resource.upper()}_Memory_Usage.png")


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
    plt.savefig("avg_latency_vs_concurrency.png")


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
    plt.savefig("throughput_vs_concurrency.png")


def parse_arguments():
    """
    Parse command-line arguments.

    Args:

    Returns:
        ArgumentParser
    """
    parser = argparse.ArgumentParser(description="Benchmarking tool for the service.")
    parser.add_argument(
        "--concurrency_levels",
        type=int,
        nargs="+",
        required=True,
        help="List of concurrency levels to test (e.g., 1 2 3 4).",
    )
    parser.add_argument(
        "--num_requests",
        type=int,
        required=True,
        help="Number of requests per concurrency level.",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=[],
        help=' list of prompts (e.g., ["Prompt 1", "Prompt 2"]).',
    )
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="URL of the service to benchmark (e.g., http://127.0.0.1:8000/query).",
    )
    return parser.parse_args()


def save_detailed_benchmark_results_to_csv(filename, benchmark_results):
    """
    Save detailed benchmark results to a CSV file.

    Args:
        filename (str): The name of the CSV file where the results will be saved.
        benchmark_results (list): List of tuples (concurrency level, detailed results dictionary).
    Return

    """
    headers = [
        "Concurrency Level",
        "Total Requests",
        "Successful Requests",
        "Failed Requests",
        "Average Latency (s)",
        "Min Latency (s)",
        "Max Latency (s)",
        "Throughput (req/s)",
        "GPU Stats Before",
        "GPU Stats After",
        "CPU Stats Before",
        "CPU Stats After",
    ]

    with open(filename, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)

        writer.writeheader()

        for concurrency, result in benchmark_results:
            writer.writerow(
                {
                    "Concurrency Level": concurrency,  # Extracted from the tuple
                    "Total Requests": result["total_requests"],
                    "Successful Requests": result["successful_requests"],
                    "Failed Requests": result["failed_requests"],
                    "Average Latency (s)": f"{result['avg_latency']:.2f}",
                    "Min Latency (s)": f"{result['min_latency']:.2f}",
                    "Max Latency (s)": f"{result['max_latency']:.2f}",
                    "Throughput (req/s)": f"{result['throughput']:.2f}",
                    "GPU Stats Before": json.dumps(result["gpu_stats_before"]),
                    "GPU Stats After": json.dumps(result["gpu_stats_after"]),
                    "CPU Stats Before": json.dumps(result["cpu_stats_before"]),
                    "CPU Stats After": json.dumps(result["cpu_stats_after"]),
                }
            )


if __name__ == "__main__":
    args = parse_arguments()
    concurrency_levels = args.concurrency_levels
    num_requests_per_level = args.num_requests
    prompts = args.prompts
    url = args.url

    results = []
    for concurrency in concurrency_levels:
        logging.debug(f"\nTesting with {concurrency} concurrent requests:")
        result = asyncio.run(
            benchmark_service(
                url,
                concurrent_requests=concurrency,
                num_requests=num_requests_per_level,
            )
        )
        results.append((concurrency, result))
    concurrency = [r[0] for r in results]
    avg_latency = [r[1]["avg_latency"] for r in results]
    throughput = [r[1]["throughput"] for r in results]
    gpu_stats = {
        r[1]["concurrent_requests"]: r[1]["gpu_stats_over_time"] for r in results
    }
    cpu_stats = {
        r[1]["concurrent_requests"]: r[1]["cpu_stats_over_time"] for r in results
    }

    save_detailed_benchmark_results_to_csv("detailed_benchmark_results.csv", results)

    plot_avg_latency_vs_concurrency(avg_latency, concurrency)
    plot_throughput_vs_concurrency(throughput, concurrency)
    plot_resource_stats_over_concurrency(gpu_stats, "gpu")
    plot_resource_stats_over_concurrency(cpu_stats, "cpu")
