import pytest
from src.core.model_manager import ModelManager
from src.core.abuse_detector import AbuseDetector

_model_manager_instance = None

@pytest.fixture(scope="session")
async def setup_model_manager():
    global _model_manager_instance
    if _model_manager_instance is None:
        _model_manager_instance = ModelManager(AbuseDetector())
        print("Model loading started.")
        await _model_manager_instance.load_model_async()
        print("Model loading finished.")
    return _model_manager_instance
