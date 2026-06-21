"""
detector_lstm.py  --  LSTM autoencoder detector (Keras training).

A sequence autoencoder. Unlike the dense AE (which treats a window as an
unordered bag of 10 numbers), the LSTM processes the gaps IN ORDER and
carries memory across timesteps. This lets it model temporal patterns
(colored jitter, thermal trend) and -- the point of this experiment --
catch PATTERN-shaped faults: slow-onset jumps, oscillatory wobble, and
frequency steps, which a single-gap threshold is structurally bad at.

Architecture
------------
    input  : sequence of SEQ_LEN gaps  (shape SEQ_LEN x 1)
    encoder: LSTM(HID) -> a single context vector (the bottleneck)
    repeat : context vector copied SEQ_LEN times
    decoder: LSTM(HID) over the repeated context
    output : TimeDistributed Dense(1) -> reconstructed sequence

How detection works
-------------------
Train ONLY on clean sequences. Normal sequences reconstruct well (low
error). A sequence containing a pattern fault reconstructs badly (high
error). Flag when per-sequence reconstruction MSE exceeds a threshold
learned from clean validation data.

Normalization: z = (gap - NOMINAL) / SCALE, same as the dense AE, so the
network sees deviations rather than the constant 1.0 baseline.

Outputs
-------
    lstm_ae.keras       trained model
    lstm_meta.npz       NOMINAL, SCALE, THRESHOLD, SEQ_LEN
"""

import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow import keras
from data_loader import load_csv

SEQ_LEN = 20        # sequence length (longer than dense W=10: pattern faults
                    # like oscillations and onsets span many frames)
HID     = 16        # LSTM hidden units (the bottleneck context vector)
NOMINAL = 1.0
EPOCHS  = 15
BATCH   = 128
STRIDE  = 5         # step between training sequences (overlap, but not 1 --
                    # keeps training set size manageable for the LSTM)
SEED    = 0

tf.random.set_seed(SEED)
np.random.seed(SEED)


def make_sequences(gaps, seq_len, stride=1):
    """Slice gaps into (N, seq_len, 1) sequences; centers = last index."""
    n = len(gaps)
    if n < seq_len:
        return np.empty((0, seq_len, 1)), np.empty((0,), dtype=int)
    starts = np.arange(0, n - seq_len + 1, stride)
    seqs = np.stack([gaps[s:s + seq_len] for s in starts])
    centers = starts + seq_len - 1
    return seqs[:, :, None], centers


def build_model():
    """LSTM autoencoder: encode to a context vector, decode back."""
    m = keras.Sequential([
        keras.layers.Input(shape=(SEQ_LEN, 1)),
        keras.layers.LSTM(HID, name="encoder"),            # -> (HID,)
        keras.layers.RepeatVector(SEQ_LEN),                # -> (SEQ_LEN, HID)
        keras.layers.LSTM(HID, return_sequences=True, name="decoder"),
        keras.layers.TimeDistributed(keras.layers.Dense(1)),
    ])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return m


def recon_error(model, X):
    """Per-sequence mean-squared reconstruction error."""
    R = model.predict(X, verbose=0)
    return ((R - X) ** 2).mean(axis=(1, 2))


if __name__ == "__main__":
    train = load_csv("dataset/train_normal.csv")
    val   = load_csv("dataset/val_normal.csv")

    SCALE = float(np.std(train["gaps"] - NOMINAL))
    print(f"Normalization: NOMINAL={NOMINAL}, SCALE={SCALE:.6e}")

    def prep(gaps):
        seqs, ctr = make_sequences((gaps - NOMINAL) / SCALE, SEQ_LEN, STRIDE)
        return seqs, ctr

    Xtr, _ = prep(train["gaps"])
    Xva, _ = prep(val["gaps"])
    print(f"train seqs: {Xtr.shape}   val seqs: {Xva.shape}")

    model = build_model()
    model.summary()
    hist = model.fit(
        Xtr, Xtr,
        validation_data=(Xva, Xva),
        epochs=EPOCHS, batch_size=BATCH, verbose=2,
    )
    print(f"final train loss: {hist.history['loss'][-1]:.6e}")
    print(f"final val   loss: {hist.history['val_loss'][-1]:.6e}")

    err_va = recon_error(model, Xva)
    k = 6.0
    THRESHOLD = float(err_va.mean() + k * err_va.std())
    fp = int((err_va > THRESHOLD).sum())
    print(f"clean val recon error: mean={err_va.mean():.3e} std={err_va.std():.3e}")
    print(f"threshold (mean+{k:.0f}std): {THRESHOLD:.3e}")
    print(f"false positives on clean val: {fp} / {len(err_va)}")

    model.save("lstm_ae.keras")
    np.savez("lstm_meta.npz", NOMINAL=NOMINAL, SCALE=SCALE,
             THRESHOLD=THRESHOLD, SEQ_LEN=SEQ_LEN, STRIDE=STRIDE)
    print("saved lstm_ae.keras + lstm_meta.npz")
