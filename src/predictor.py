import joblib

# Load trained model
model = joblib.load("models/breast_cancer_model.pkl")


def predict(data):
    prediction = model.predict(data)
    probability = model.predict_proba(data)

    return prediction, probability

def get_model():
    return model