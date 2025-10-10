from sklearn.ensemble import RandomForestClassifier
import joblib
from data import load_data, split_data

def fit_model(X_train, y_train):
    """
    Train a Random Forest Classifier on the training data 
    and save the trained model as a pickle file.
    
    Args:
        X_train (numpy.ndarray): Training feature set.
        y_train (numpy.ndarray): Training labels.
    """
     
    clf = RandomForestClassifier(n_estimators=100, random_state=12)
    clf.fit(X_train, y_train)
    joblib.dump(clf, "../model/breast_cancer_model.pkl")

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    fit_model(X_train, y_train)
