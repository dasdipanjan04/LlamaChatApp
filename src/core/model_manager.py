"""Loads model, generates response from llama model."""

import os
import torch
from threading import Lock
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)
from asyncio import to_thread
from huggingface_hub import HfFolder
from threading import Thread
from src.core.quantization import get_quantization_config
from src.core.abuse_detector import AbuseDetector

load_dotenv()
os.environ["CURL_CA_BUNDLE"] = ""
api_token = os.getenv("API_TOKEN")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

HfFolder.save_token(api_token)


class ModelManager:
    """
    The ModelManager class handles the following:
    * Load the model using Bitandbytes 4 bit Qunatisation.
    * Load model asyncly
    * Checks whether the prompt is abusive or not
    * Finally Generates Response
    """

    def __init__(self, abuse_detector: AbuseDetector):
        self.model = None
        self.tokenizer = None
        self.abuse_detector = abuse_detector
        self.lock = Lock()

    def _load_model(self):
        """
        Loads the meta-llama/Llama-3.1-8B model from hugging face
        Uses a 4 bit quantiosation config to load model in 4 bit
        Initialize the abuse detection pipeline as well

        Args:
            self

        Returns:
            None
        """
        llm_model = "meta-llama/Llama-3.1-8B"

        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_model,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
                self.model.resize_token_embeddings(len(self.tokenizer))

        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model, quantization_config=get_quantization_config(), device_map="auto"
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()

    async def load_model_async(self):
        """
        Loads the model asyncly

        Args:
            self

        Returns:
            None
        """
        with self.lock:
            if self.model is None or self.abuse_detector is None:
                await to_thread(self._load_model)

    async def generate_response(self, text: str):
        """
        Generates the response from the llama model.

        Args:
            self
            text (str): The text or prompt which will be passed to the llama model

        Returns:
            None
        """
        await self.load_model_async()

        if self.abuse_detector.is_abusive(text):
            yield "Please refrain from such language.\nLet us have a constructive conversation."
            return

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True,
            return_attention_mask=True,
        ).to("cuda")
        streamer = TextIteratorStreamer(
            self.tokenizer,
            timeout=60.0,
            skip_special_tokens=True,
            skip_prompt=True,
        )

        generate_kwargs = dict(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=100,
            min_length=10,
            length_penalty=0.5,
            do_sample=True,
            top_p=0.8,
            top_k=20,
            temperature=0.5,
            streamer=streamer,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=self.tokenizer.pad_token_id,
            early_stopping=True,
        )

        t = Thread(target=self.model.generate, kwargs=generate_kwargs)
        t.start()

        for token in streamer:
            yield token
