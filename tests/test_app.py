import sys, os
import pytest
import torch
from unittest.mock import MagicMock

# --- Create dummy replacements ---
class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    chat_template = "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}{% endfor %}"

    def apply_chat_template(self, conv, tokenize=False, add_generation_prompt=True):
        return "mock_prompt"

    def __call__(self, prompt, return_tensors=None):
        # Must look like transformers output
        return {"input_ids": torch.tensor([[0, 1]])}

    def decode(self, tokens, skip_special_tokens=True):
        return "• Easy switch"


class DummyModel:
    def to(self, device): return self
    def eval(self): return self
    def generate(self, **kwargs): return torch.tensor([[0, 1, 2, 3]])


# --- Insert mocks into sys.modules before app is imported ---
sys.modules["transformers"] = MagicMock(
    AutoTokenizer=MagicMock(from_pretrained=lambda *a, **kw: DummyTokenizer()),
    AutoModelForCausalLM=MagicMock(from_pretrained=lambda *a, **kw: DummyModel())
)

# Now safe to import app
import app


# ================== Tests ==================

def test_calculate_footprint():
    total, stats = app.calculate_footprint(
        car_km=10, bus_km=5, train_km=2, air_km_week=50,
        meat_meals=3, vegetarian_meals=2, vegan_meals=1,
    )
    assert total > 0
    assert "trees" in stats
    assert isinstance(stats["trees"], int)


def test_chat_mock_runs():
    out = app.chat(messages=[{"role": "user", "content": "Hello"}], history=[])
    assert isinstance(out, str)
    assert "Easy switch" in out or "•" in out
