import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app

def test_calculate_footprint():
    """Check CO2 footprint calculation."""
    total, stats = app.calculate_footprint(
        car_km=10, bus_km=5, train_km=2, air_km_week=50,
        meat_meals=3, vegetarian_meals=2, vegan_meals=1,
    )

    assert total > 0
    assert "trees" in stats
    assert isinstance(stats["trees"], int)
