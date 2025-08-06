
from typing import Tuple
import numpy as np

__all__ = ["generate_people_data", "filter_people"]

def generate_people_data(
    num_samples: int = 100,
    age_range: Tuple[int, int] = (18, 70),
    income_range: Tuple[int, int] = (20000, 120000)
) -> np.ndarray:
    """
    Generate synthetic people data with age, income, purchase history, and frequency.
    Returns: np.ndarray of shape (num_samples, 4)
    """
    ages = np.random.randint(age_range[0], age_range[1]+1, num_samples)
    incomes = np.random.randint(income_range[0], income_range[1]+1, num_samples)
    purchase_history = np.random.randint(100, 50000, num_samples)
    frequency = np.random.randint(1, 100, num_samples)
    data = np.column_stack((ages, incomes, purchase_history, frequency))
    return data

def filter_people(
    people_data: np.ndarray,
    age_range: Tuple[int, int] = (25, 35),
    high_income_threshold: int = 100000,
    high_frequency_threshold: int = 80
) -> np.ndarray:
    """
    Filter people by age, income, and frequency thresholds.
    Returns: np.ndarray of filtered people
    """
    mask = (
        (people_data[:, 0] >= age_range[0]) & (people_data[:, 0] <= age_range[1]) &
        (people_data[:, 1] >= high_income_threshold) &
        (people_data[:, 3] >= high_frequency_threshold)
    )
    return people_data[mask]
