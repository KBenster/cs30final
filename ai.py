import tensorflow as tf
import dataset_interpreter
import numpy as np

features, labels = dataset_interpreter.get_sentiment_features_labels()

train_dataset = tf.data.Dataset.from_tensor_slices((features, labels))
#should be <_PrefetchDataset element_spec=(TensorSpec(shape=(), dtype=tf.string, name=None), TensorSpec(shape=(), dtype=tf.int64, name=None))>
# is       <_TensorSliceDataset element_spec=(TensorSpec(shape=(), dtype=tf.string, name=None), TensorSpec(shape=(1,), dtype=tf.string, name=None))>
print(train_dataset)
#test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

BUFFER_SIZE = 9000
BATCH_SIZE = 64

train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

VOCAB_SIZE = 1000
encoder = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE
)
encoder.adapt(train_dataset.map(lambda text, label: text))

vocab = np.array(encoder.get_vocabulary())

model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        mask_zero=True
    ),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model.fit(features, labels, epochs=3)

model.save("financialtest1", save_format="tf")
# model.load_weights("goodmodel")

sample_text = "company reported net loss and bankrupty. my wife left me. loss $4 billion."
predictions = model.predict(np.array([sample_text]))

print(predictions)