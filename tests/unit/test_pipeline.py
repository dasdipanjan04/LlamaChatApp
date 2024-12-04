import re
import pytest
from src.pipeline import ModelManager


@pytest.mark.usefixtures("setup_model_manager")
def test_model_initialisation(setup_model_manager: ModelManager):
    assert setup_model_manager.model is not None, "Model should be initialised"
    assert setup_model_manager.tokenizer is not None, "Tokeniser should be initialised"


@pytest.mark.usefixtures("setup_model_manager")
def test_abuse_detection_positive(setup_model_manager: ModelManager):
    offensive_text = "You are stupid and dumb."
    is_abusive = setup_model_manager.is_abusive(offensive_text)
    assert is_abusive is True, "Abuse should be detected!"


@pytest.mark.usefixtures("setup_model_manager")
def test_abuse_detection_negative(setup_model_manager: ModelManager):
    non_offensive_text = "Who is the author of the book 'How to kill a mocking bird'"
    is_non_abusive = setup_model_manager.is_abusive(non_offensive_text)
    assert (
        is_non_abusive is False
    ), "Non offensive sentences should not be detected as abusive!"


@pytest.mark.usefixtures("setup_model_manager")
@pytest.mark.asyncio
async def test_response_generation(setup_model_manager: ModelManager):
    input_text = "What is the capital city of Germany?"
    response = ""
    try:
        async for token in setup_model_manager.generate_response(input_text):
            response += token
    except Exception as e:
        pytest.fail(f"Response generation failed: {e}")

    assert response, "Response should not be empty"
    assert re.search(
        r"\bBerlin\b", response, re.IGNORECASE
    ), "Response should mention Berlin."
