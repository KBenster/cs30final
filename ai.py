import tensorflow as tf
import dataset_interpreter
import numpy as np

labels, features = dataset_interpreter.get_sentiment_features_labels()
features = features.reshape(features.shape[0])

print(labels.take(1), features.take(1))

train_dataset = tf.data.Dataset.from_tensor_slices((features, labels))
print("a", train_dataset.take(1))
#^ this prints
# <_TakeDataset element_spec=(
#     TensorSpec(shape=(), dtype=tf.int16, name=None), 
#     TensorSpec(shape=(1,), dtype=tf.string, name=None))>

BUFFER_SIZE = 9000
BATCH_SIZE = 64

train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

VOCAB_SIZE = 50000
encoder = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE
)

encoder.adapt(features)

#rank 0 = scalar
#rank 1 = vector
#rank 2 = N x N matrix
#rank >=3 = tensor

vocab = np.array(encoder.get_vocabulary())

model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        mask_zero=True
    ),
    #LSTM has more parameters and flexibility than GRU, but has a higher risk of overfitting
    #and has a higher computational cost. GRU has fewer parameters
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(2048)),
    tf.keras.layers.Dense(128, activation='sigmoid'),
    tf.keras.layers.Dense(128, activation='sigmoid'),
    tf.keras.layers.Dense(1, activation='elu')
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model.fit(train_dataset, epochs=1)

model.save("financialtest2", save_format="tf")
#model.load_weights("financialtest1")

sample_text = "company reported net loss and bankrupty. my wife left me. loss $4 billion."
predictions = model.predict(np.array([sample_text]))

print(predictions)

while True:
    t = input()
    predictions = model.predict(np.array([t]))
    print(predictions)