import numpy as np
from tensorflow import keras
from data_loader import load_csv
import tensorflow as tf
tf.random.set_seed(0)
np.random.seed(0)
SEQ_LEN = 20

# Load data
train = load_csv("dataset/train_normal.csv")
val = load_csv("dataset/val_normal.csv")

# Normalize
nominal = 1.0
scale = np.std(train["gaps"] - nominal)

# Create sequences
def make_seq(gaps):
    X = []
    for i in range(0, len(gaps)-SEQ_LEN+1, 5):
        X.append(gaps[i:i+SEQ_LEN])
    return np.array(X).reshape(-1, SEQ_LEN, 1)

train_gaps = (train["gaps"] - nominal) / scale
val_gaps = (val["gaps"] - nominal) / scale

X_train = make_seq(train_gaps)
X_val = make_seq(val_gaps)

# LSTM Autoencoder
model = keras.Sequential([
    keras.layers.Input(shape=(SEQ_LEN,1)),
    keras.layers.LSTM(16),
    keras.layers.RepeatVector(SEQ_LEN),
    keras.layers.LSTM(16, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(1))
])

model.compile(optimizer="adam", loss="mse")

# Train
model.fit(
    X_train,
    X_train,
    epochs=15,
    batch_size=128,
    validation_data=(X_val, X_val)
)

# Reconstruction error
pred = model.predict(X_val)

error = np.mean((pred - X_val)**2, axis=(1,2))

threshold = error.mean() + 6 * error.std()

print("Threshold =", threshold)

fp = np.sum(error > threshold)
print("False positives:", fp, "/", len(error))

model.save("lstm_ae.keras")
np.savez("lstm_meta.npz", NOMINAL=nominal, SCALE=scale,
         THRESHOLD=threshold, SEQ_LEN=SEQ_LEN, STRIDE=1)