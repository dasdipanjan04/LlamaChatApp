"""Loads model, generates response from llama model."""

import os
import torch
import json
from threading import Lock
from pathlib import Path
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    StoppingCriteriaList,
)
from asyncio import to_thread
from huggingface_hub import HfFolder
from threading import Thread
from src.core.quantization import get_quantization_config
from src.core.abuse_detector import AbuseDetector
from src.core.multi_sentence_stopping_criteria import MultiSentenceStoppingCriteria

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

    def __init__(
        self,
        abuse_detector: AbuseDetector,
        llm_model: str = "mistralai/Mistral-7B-v0.1",
    ):
        self.model = None
        self.tokenizer = None
        self.abuse_detector = abuse_detector
        self.llm_model = llm_model
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

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.llm_model,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            padding_side="left",
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.llm_model,
            quantization_config=get_quantization_config(),
            device_map="auto",
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = (
                self.tokenizer.eos_token_id
                or self.tokenizer.convert_tokens_to_ids("<pad>")
            )
            self.model.resize_token_embeddings(len(self.tokenizer))

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

        base_dir = Path(__file__).resolve().parent.parent.parent
        config_path = base_dir / "config.json"

        with open(config_path, "r") as config_file:
            config = json.load(config_file)

        inputs = self.tokenizer(text, **config["tokenizer"]).to("cuda")

        streamer = TextIteratorStreamer(self.tokenizer, **config["streamer"])
        stopping_criteria = StoppingCriteriaList(
            [
                MultiSentenceStoppingCriteria(
                    self.tokenizer, **config["stopping_criteria"], punctuations=[".", "!", "?"]
                )
            ]
        )
        generate_kwargs = dict(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **config["generate_kwargs"],
            streamer=streamer,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            stopping_criteria=stopping_criteria,
        )

        t = Thread(target=self.model.generate, kwargs=generate_kwargs)
        t.start()

        for token in streamer:
            yield token
