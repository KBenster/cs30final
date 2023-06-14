import tensorflow as tf
import requests
import pandas as pd
import numpy as np
import json
import csv

def get_data(post_limit, comment_limit):

    data = {}

    #public reddit API
    response = requests.get('https://www.reddit.com/r/wallstreetbets/hot/.json', 
                            headers={'User-Agent': 'Mozilla/5.0'})

    if response.status_code == 200:
        data = response.json()

        #the json data contains all of the posts
        posts = data['data']['children']

        #iterate over top posts
        for post in posts[2:post_limit+2]:
            post_data = post['data']
            post_text = ""
            post_text += post_data['title']
            post_text += post_data['selftext']
            # print(post_data['title'])
            # print(post_data['selftext'])

            #get comments
            post_url = 'https://www.reddit.com' + post_data['permalink'] + '.json'
            response_comments = requests.get(post_url,
                                            headers={'User-Agent': 'Mozilla/5.0'})

            if response_comments.status_code == 200:
                comments_data = response_comments.json()
                comments = comments_data[1]['data']['children']

                for comment in comments[1:comment_limit+1]: # remove the auto moderation comment
                    post_text += comment['data']['body']
                    #print("comment", comment['data']['body'])
            else:
                print(response_comments.status_code)
            data[post_data['permalink']] = post_text
    else:
        print(response.status_code)
    return data

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
    sentiment_labels = sentiment_labels.astype(np.int16)

    return sentiment_labels, sentiment_features
# https://www.tensorflow.org/tutorials/load_data/csv

labels, features = get_sentiment_features_labels()
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
        output_dim=128,
        mask_zero=True
    ),
    #LSTM has more parameters and flexibility than GRU, but has a higher risk of overfitting
    #and has a higher computational cost. GRU has fewer parameters
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),  #return_sequences=True will make the output of this layer pass into the next LSTM layer
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
    tf.keras.layers.Dense(256, activation='sigmoid'),
    tf.keras.layers.Dense(128, activation='sigmoid'),
    tf.keras.layers.Dense(1, activation='tanh')

    #elu activation goes logarithmically to -1 and exponentially positive
    #sigmoid goes from -1 to 1
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

#history = model.fit(train_dataset, epochs=5)

#model.save("saved_model", save_format="tf")
model.load_weights("saved_model").expect_partial() # expect_partial hides some nonsense error codes

# sample_text = "company reported net loss and bankrupty. my wife left me. loss $4 billion."
# predictions = model.predict(np.array([sample_text]))
#print(predictions)
data = get_data(10, 10)
keys = list(data.keys())
new_data = {}
for i in range(len(keys))[::-1][:-2]:
    key = keys[i]
    if type(data[key]) == dict:
        data.pop(key)
    #print(data[key])
    print(key, data[key])
    new_data[key] = float(model.predict([data[key]])[0][0])

with open("serve/output.json", "w") as file:
    json.dump(new_data, file)