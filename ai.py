import tensorflow as tf
import dataset_interpreter

features, labels = dataset_interpreter.get_sentiment_features_labels()

print(type(features), type(labels))