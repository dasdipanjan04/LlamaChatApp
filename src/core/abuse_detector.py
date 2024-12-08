import torch
from transformers import pipeline


class AbuseDetector:
    def __init__(self, model_name="unitary/toxic-bert"):
        self.abuse_detector = pipeline(
            "text-classification",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1,
        )

    def is_abusive(self, text: str) -> bool:
        """
        Checks whether a certain prompt or response is abusive or not.

        Args:
            self
            text (str): The text which needs to checked using the abuse detector

        Returns:
            True or False based on whether the text is abusive or not.
        """
        if not self.abuse_detector:
            raise ValueError("Abuse detector is not initialized.")
        results = self.abuse_detector(text)
        return any(
            result["label"] in ["toxic", "severe_toxic", "offensive"]
            and result["score"] > 0.5
            for result in results
        )
