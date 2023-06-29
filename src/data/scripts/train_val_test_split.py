import numpy as np
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Load data
    data = np.load("data/final/data.npy")

    # Split data
    n = len(data)
    data_train, data_test = train_test_split(data, test_size=int(n * .15), random_state=42, shuffle=True,
                                             stratify=data[:, 1])
    data_train, data_val = train_test_split(data_train, test_size=int(n * .10), random_state=42, shuffle=True,
                                            stratify=data_train[:, 1])

    # Save splits
    np.save("data/final/data_train.npy", data_train)
    np.save("data/final/data_val.npy", data_val)
    np.save("data/final/data_test.npy", data_test)
