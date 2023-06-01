import tensorflow as tf
import pandas as pd
import numpy as np
import csv

def get_sentiment_features_labels():
    sentiment_train = pd.read_csv(
        "./datasets/financialsentiment2.csv",
        names=["text", "sentiment"],
        encoding='latin-1' # oooga booga utf-8
    )

    sentiment_features = sentiment_train.copy()
    sentiment_labels = sentiment_features.pop('sentiment')

    sentiment_features = np.array(sentiment_features)
    sentiment_labels = np.array(sentiment_labels)
    sentiment_labels[sentiment_labels == ''] = 0.0
    sentiment_labels = sentiment_labels.astype(np.int16)

    return sentiment_labels, sentiment_features
# https://www.tensorflow.org/tutorials/load_data/csv

def get_twitter_features_labels():
    return None