import torch
from transformers import StoppingCriteria
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt_tab")


class MultiSentenceStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, num_sentences=3, punctuations=None):
        """
        Initialize the stopping criteria to allow a specified number of sentences.

        Args:
            tokenizer: The tokenizer of the model.
            num_sentences: Number of sentences before stopping.
            punctuations: List of sentence-ending punctuations.
        """
        if punctuations is None:
            punctuations = [".", "!", "?"]
        self.tokenizer = tokenizer
        self.eos_tokens = [tokenizer.convert_tokens_to_ids(p) for p in punctuations]
        self.num_sentences = num_sentences
        self.current_text = ""

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        """
        Check if the stopping criterion is met.

        Args:
            input_ids: Current input IDs during generation.
            scores: I have not used the score

        Return:
            bool: Whether generation should stop or not.
        """
        new_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

        self.current_text += new_text

        sentences = sent_tokenize(self.current_text)
        return len(sentences) >= self.num_sentences
