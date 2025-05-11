# utils.py
import joblib, os

def save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)

def load(path):
    return joblib.load(path)
