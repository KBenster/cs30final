import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import os
import sqlite3

#disable tqdm progress bar
tfds.disable_progress_bar()

database_connection = sqlite3.connect('database.db')
# database_connection.execute("""CREATE TABLE `sentiments` (
#                                   `ticker` TEXT(200) NOT NULL,
#                                   `sentiment` INT
#                                 );""")


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


for example, label in train_dataset.take(1):
  print('text: ', example.numpy())
  print('label: ', label.numpy())

BUFFER_SIZE = 10000
BATCH_SIZE = 64

#Shuffle() shuffles the elements in the dataset. It will select BUFFER_SIZE elements and shuffle them
#returns the new dataset with the shuffled elements

#Batch() will divide the dataset into batches of BATCH_SIZE

train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# for example, label in train_dataset.take(1):
#   print('texts: ', example.numpy()[:3])
#   print()
#   print('labels: ', label.numpy()[:3])


#The adapt method sets the layer's vocabulary
VOCAB_SIZE = 1000
encoder = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset.map(lambda text, label: text))

vocab = np.array(encoder.get_vocabulary())

# encoded_example = encoder(example)[:3].numpy()
# print(encoded_example)

# for n in range(3):
#   print("Original: ", example[n].numpy())
#   print("Round-trip: ", " ".join(vocab[encoded_example[n]]))
#   print()
     
model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

#print([layer.supports_masking for layer in model.layers])



model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

#saving and checkpoints
checkpoint_path = "traininghistory/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

#train the model...
# history = model.fit(train_dataset, epochs=5,
#                     validation_data=test_dataset,
#                     validation_steps=30,
#                     callbacks=[cp_callback])

#model.save("model1.h5", save_format='tf')
model.load_weights("model1.h5", load_format='tf')
test_loss, test_acc = model.evaluate(test_dataset)

print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

# plt.figure(figsize=(16, 8))
# plt.subplot(1, 2, 1)
# plot_graphs(history, 'accuracy')
# plt.ylim(None, 1)
# plt.subplot(1, 2, 2)
# plot_graphs(history, 'loss')
# plt.ylim(0, None)

sample_text = ('The movie was cool. The animation and the graphics '
               'were out of this world. I would recommend this movie.')
predictions = model.predict(np.array([sample_text]))
print(predictions)
database_connection.close()

# ipynb version
# https://github.com/tensorflow/text/blob/master/docs/tutorials/text_classification_rnn.ipynb

# https://www.tensorflow.org/text/tutorials/text_classification_rnn
# https://www.tensorflow.org/api_docs/python/tf/data/Dataset#methods_2

# saving models so you dont have to retrain every time you run the program
# https://www.tensorflow.org/tutorials/keras/save_and_load


#SETUP TOOLS INFORMATION
#INSHALLAH WE WILL FIX THIS
# PS C:\Users\1302215\Desktop\github\cs30final> python -m pip show setuptools
# Name: setuptools
# Version: 65.5.0
# Summary: Easily download, build, install, upgrade, and uninstall Python packages
# Home-page: https://github.com/pypa/setuptools
# Author: Python Packaging Authority
# Author-email: distutils-sig@python.org
# License:
# Location: c:\program files\windowsapps\pythonsoftwarefoundation.python.3.10_3.10.3056.0_x64__qbz5n2kfra8p0\lib\site-packages
# Requires:
# Required-by: tensorboard