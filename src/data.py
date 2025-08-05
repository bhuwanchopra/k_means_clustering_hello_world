import numpy as np

def generate_people_data(num_samples=100, age_range=(18, 70), income_range=(20000, 120000)):
    ages = np.random.randint(age_range[0], age_range[1]+1, num_samples)
    incomes = np.random.randint(income_range[0], income_range[1]+1, num_samples)
    purchase_history = np.random.randint(100, 50000, num_samples)
    frequency = np.random.randint(1, 100, num_samples)
    data = np.column_stack((ages, incomes, purchase_history, frequency))
    return data

def filter_people(people_data, age_range=(25, 35), high_income_threshold=100000, high_frequency_threshold=80):
    mask = (
        (people_data[:, 0] >= age_range[0]) & (people_data[:, 0] <= age_range[1]) &
        (people_data[:, 1] >= high_income_threshold) &
        (people_data[:, 3] >= high_frequency_threshold)
    )
    return people_data[mask]
