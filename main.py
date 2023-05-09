import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

#disable tqdm progress bar
tfds.disable_progress_bar()

def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])

dataset, info = tfds.load('imdb_reviews', with_info=True,
                          as_supervised=True)

train_dataset, test_dataset = dataset['train'], dataset['test']

#attribute
#The type specification of an element in this dataset
#train_dataset.element_spec
#would return
# (TensorSpec(shape=(), dtype=tf.string, name=None),
#  TensorSpec(shape=(), dtype=tf.int64, name=None))


# for example, label in train_dataset.take(1):
#   print('text: ', example.numpy())
#   print('label: ', label.numpy())

BUFFER_SIZE = 10000
BATCH_SIZE = 64

https://www.tensorflow.org/text/tutorials/text_classification_rnn
https://www.tensorflow.org/api_docs/python/tf/data/Dataset#methods_2