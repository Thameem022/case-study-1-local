import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
from unittest.mock import patch, MagicMock

@pytest.fixture(autouse=True, scope="session")
def patch_hf_classes():
    with patch("app.AutoTokenizer") as mock_tokenizer_cls, \
         patch("app.AutoModelForCausalLM") as mock_model_cls:

        # Fake tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.chat_template = "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}{% endfor %}"
        mock_tokenizer.apply_chat_template.return_value = "mock_prompt"
        mock_tokenizer.__call__.return_value = {"input_ids": [[0, 1]]}
        mock_tokenizer.decode.return_value = "• Easy switch"

        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.generate.return_value = [[0, 1, 2, 3]]
        mock_model_cls.from_pretrained.return_value = mock_model

        yield 

import app


def test_calculate_footprint():
    total, stats = app.calculate_footprint(
        car_km=10, bus_km=5, train_km=2, air_km_week=50,
        meat_meals=3, vegetarian_meals=2, vegan_meals=1,
    )
    assert total > 0
    assert "trees" in stats
    assert isinstance(stats["trees"], int)


def test_chat_mock_runs():
    """chat() should return mocked text without loading real model."""
    out = app.chat(messages=[{"role": "user", "content": "Hello"}], history=[])
    assert isinstance(out, str)
    assert "Easy switch" in out or "•" in out
