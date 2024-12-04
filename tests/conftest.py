import pytest
from src.pipeline import model_manager


@pytest.fixture(scope="session")
async def setup_model_manager():
    print("Model loading started.")
    await model_manager.load_model_async()
    print("Model loading finished.")
    return model_manager
