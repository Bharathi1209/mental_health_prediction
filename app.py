from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Bidirectional, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# ------------------------------------------------
# SAME CODE YOU ALREADY HAVE (no internal changes)
# ------------------------------------------------

df = pd.read_csv(r"C:\Users\BHARATHI\Downloads\data_to_be_cleansed.csv\data_to_be_cleansed.csv")
texts = df["text"].astype(str).values
labels = df["target"].values

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

texts = [clean_text(t) for t in texts]

encoder = LabelEncoder()
labels = encoder.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

max_words = 10000
max_len = 100

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding="post", truncating="post")
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding="post", truncating="post")

model = Sequential([
    Embedding(max_words, 128, input_length=max_len),
    Conv1D(filters=128, kernel_size=5, activation="relu"),
    MaxPooling1D(pool_size=2),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(5, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

model.fit(
    X_train_pad, y_train,
    epochs=3,  # reduce for quick testing
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop],
    class_weight=class_weights,
    verbose=1
)

disorders = {
    0: "Stress",
    1: "Depression",
    2: "Bipolar disorder",
    3: "Personality disorder",
    4: "Anxiety"
}

def predict_sentence(sentence):
    seq = tokenizer.texts_to_sequences([clean_text(sentence)])
    pad = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    probs = model.predict(pad, verbose=0)[0]
    pred = np.argmax(probs)
    return probs, pred, disorders[pred]

# ------------------------------------------------
# FLASK APP
# ------------------------------------------------
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probs = None
    if request.method == "POST":
        user_input = request.form["sentence"]
        probs, pred_idx, pred_label = predict_sentence(user_input)
        prediction = pred_label
        probs = {disorders[i]: f"{p:.2f}" for i, p in enumerate(probs)}
    return render_template("index.html", prediction=prediction, probs=probs)

if __name__ == "__main__":
    app.run(debug=True)
