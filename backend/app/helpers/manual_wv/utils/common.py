import os
import pandas as pd
import joblib
import tensorflow as tf


def load_parquet(path):
    path = path + ".parquet"
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


def load_csv(path):
    path = path + ".csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def load_model(path):
    path = path + ".keras"
    if os.path.exists(path):
        return tf.keras.models.load_model(path)
    return None


def load_scaler(path):
    path = path + ".pkl"
    if os.path.exists(path):
        return joblib.load(path)
    return None


def load_iso_forest(writer_id, dir="iso_forests", sub_dir="and"):
    scaler_path = os.path.join(dir, sub_dir, f"{writer_id}_iso_forest.pkl")
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)
    return None
