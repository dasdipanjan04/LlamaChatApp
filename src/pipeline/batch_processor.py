import asyncio
import time
from queue import Queue
from threading import Thread
from transformers import TextIteratorStreamer

class BatchProcessor:
    def __init__(self, model_manager, batch_size=4, batch_timeout=0.05):
        self.worker_thread = None
        self.model_manager = model_manager
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.queue = Queue()
        self.responses = {}
        self.running = False

    async def enqueue_request(self, request_id, text):
        response_queue = asyncio.Queue()
        self.responses[request_id] = response_queue
        self.queue.put((request_id, text))
        return response_queue

    def _batch_worker(self):
        while self.running:
            batch = []
            while not self.queue.empty() and len(batch) < self.batch_size:
                batch.append(self.queue.get())

            if batch:
                request_ids, texts = zip(*batch)
                asyncio.run(self._process_batch(request_ids, texts))
            else:
                time.sleep(self.batch_timeout)

    async def _process_batch(self, request_ids, texts):
        inputs = [
            self.model_manager.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
                return_attention_mask=True,
            ).to("cuda")
            for text in texts
        ]

        streamers = [
            TextIteratorStreamer(
                self.model_manager.tokenizer,
                timeout=60.0,
                skip_special_tokens=True,
                skip_prompt=True,
            )
            for _ in texts
        ]

        threads = []
        for inp, streamer in zip(inputs, streamers):
            generate_kwargs = dict(
                input_ids=inp["input_ids"],
                attention_mask=inp["attention_mask"],
                max_new_tokens=300,
                do_sample=True,
                top_p=0.85,
                top_k=50,
                temperature=0.5,
                streamer=streamer,
                eos_token_id=self.model_manager.tokenizer.eos_token_id,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )

            threads.append(Thread(target=self.model_manager.model.generate, kwargs=generate_kwargs))

        for t in threads:
            t.start()

        for request_id, streamer in zip(request_ids, streamers):
            response_queue = self.responses[request_id]
            try:
                for token in streamer:
                    await response_queue.put(token)
                await response_queue.put(None)
            except Exception as e:
                await response_queue.put(f"An error occurred: {e}")
                await response_queue.put(None)

    def start(self):
        self.running = True
        self.worker_thread = Thread(target=self._batch_worker)
        self.worker_thread.start()

    def stop(self):
        self.running = False
        self.worker_thread.join()
