import os
import torch
from threading import Lock
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
    pipeline,
)
from asyncio import to_thread
from huggingface_hub import HfFolder
from threading import Thread

load_dotenv()

api_token = os.getenv("API_TOKEN")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

HfFolder.save_token(api_token)


class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.abuse_detector = None
        self.lock = Lock()

    async def load_model_async(self):
        with self.lock:
            if self.model is None or self.abuse_detector is None:
                await to_thread(self._load_model)

    def _load_model(self):
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

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model,
            quantization_config=quantization_config,
            device_map="auto",
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()

        self.abuse_detector = pipeline(
            "text-classification",
            model="unitary/toxic-bert",
            device=0 if torch.cuda.is_available() else -1,
        )

    def is_abusive(self, text: str) -> bool:
        if not self.abuse_detector:
            raise ValueError("Abuse detector is not initialized.")
        results = self.abuse_detector(text)
        return any(
            result["label"] in ["toxic", "severe_toxic", "offensive"]
            and result["score"] > 0.5
            for result in results
        )

    async def generate_response(self, text: str):
        await self.load_model_async()

        if self.is_abusive(text):
            yield "Please refrain from such language.\nLet us have a constructive conversation."
            return

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
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
            do_sample=True,
            top_p=0.8,
            top_k=20,
            temperature=0.5,
            streamer=streamer,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=self.tokenizer.pad_token_id,
            early_stopping=False,
        )

        t = Thread(target=self.model.generate, kwargs=generate_kwargs)
        t.start()

        for token in streamer:
            yield token


model_manager = ModelManager()
