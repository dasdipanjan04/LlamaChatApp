import asyncio


class ThreadManager:
    """Thread Manager class effectively manages the threads."""

    def __init__(self):
        self.global_thread_index = 0
        self.thread_index_lock = asyncio.Lock()

    async def get_next_thread_index(self):
        """
        Get next thread index asyncly

        Args:

        Returns:
            Returns the thread_index
        """
        async with self.thread_index_lock:
            thread_index = self.global_thread_index
            self.global_thread_index += 1
            return thread_index
