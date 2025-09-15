import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app


def test_chat_function_exists():
    """Check that the app has a chat() function."""
    assert hasattr(app, "chat")
    assert callable(app.chat)


def test_chat_returns_string():
    """chat() should return a non-empty string for minimal input."""
    result = app.chat(
        messages="I'm buying a bottle of water.",
        history=[],
        car_km=0,
        bus_km=0,
        train_km=0,
        air_km_month=0,
        meat_meals=0,
        vegetarian_meals=0,
        vegan_meals=0,
    )

    assert isinstance(result, str)
    assert len(result) > 0
