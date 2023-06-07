import tensorflow as tf
import dataset_interpreter
import numpy as np

features, labels = dataset_interpreter.get_sentiment_features_labels()

train_dataset = tf.data.Dataset.from_tensor_slices((features, labels))
print("a", train_dataset.take(1))
#^ this prints
# <_TakeDataset element_spec=(
#     TensorSpec(shape=(), dtype=tf.int16, name=None), 
#     TensorSpec(shape=(1,), dtype=tf.string, name=None))>

BUFFER_SIZE = 9000
BATCH_SIZE = 64

train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

VOCAB_SIZE = 1000
encoder = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE
)

def ohmygod(a):
    print("oh my god ", a)
    return a
encoder.adapt(train_dataset.map(lambda label, feature:ohmygod(feature))) #encoder.adapt(train_dataset.map(lambda text, label: text))
#When using `TextVectorization` to tokenize strings, 
#the input rank must be 1 or the last shape dimension must be 1. Received: inputs.shape=(None, None) with rank=2

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