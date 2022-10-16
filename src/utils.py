from pandas import DataFrame
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import pickle


def save_as_pickle(df: DataFrame, path: str) -> None:
    df.to_pickle(path)

def save_model(model, path: str) -> None:
    pickle.dump(model, open(path, 'wb'))

def save_ohe(ohe: OneHotEncoder, path: str) -> None:
    pickle.dump(ohe, open(path, 'wb'))

def save_scaler(scaler: MinMaxScaler, path: str) -> None:
    pickle.dump(scaler, open(path, 'wb'))

def load_model(path: str):
    return pickle.load(open(path, 'rb'))

def load_ohe(path: str) -> OneHotEncoder:
    return pickle.load(open(path, 'rb'))

def load_scaler(path: str) -> MinMaxScaler:
    return pickle.load(open(path, 'rb'))
     