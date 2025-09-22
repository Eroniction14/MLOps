from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def load_data():
    """
    Load Breast Cancer dataset and return only 10 selected features.
    """
    data = load_breast_cancer()
    # Select first 10 features
    X = data.data[:, :10]  
    y = data.target
    return X, y

def split_data(X, y):
    return train_test_split(X, y, test_size=0.3, random_state=12)
