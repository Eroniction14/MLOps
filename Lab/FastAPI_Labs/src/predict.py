import joblib

def predict_data(X):
    """
    Load the trained Breast Cancer model from file 
    and make predictions on new input data.
    
    Args:
        X (list or numpy.ndarray): Input feature values to predict on. 
                                   Must match the number of features used during training.
    
    Returns:
        numpy.ndarray: Predicted class labels (0 = Malignant, 1 = Benign).
    """
    model = joblib.load("../model/breast_cancer_model.pkl")
    return model.predict(X)
