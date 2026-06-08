"""
detector_autoencoder.py  --  Dense autoencoder detector (Keras training).

Part 1 of the hybrid detector. Catches PUNCTUAL anomalies: sudden jumps
and missing frames. Both appear as one abnormal gap inside a window, so
the same network catches both.

Architecture (deliberately tiny, for hand-coding in C later):
    input  10  (a window of 10 consecutive inter-frame gaps)
    hidden  5  ReLU      <-- 2x compression bottleneck
    output 10  linear    <-- reconstruction of the input window

How detection works
-------------------
Train ONLY on clean windows. The network learns to reconstruct normal
windows well (low error). A window containing a jump or missing frame is
unlike anything seen in training, so it reconstructs badly (high error).
Flag when reconstruction error (MSE over the window) exceeds a threshold
learned from clean validation data.

Normalization
-------------
Raw gaps sit near 1.0 with ~0.0002 spread. We normalize each value as
    z = (gap - NOMINAL) / SCALE
so the network sees DEVIATIONS, not the constant 1.0 offset. SCALE is the
std of clean deviations. The same two constants (NOMINAL, SCALE) must be
hard-coded into the C version for the math to match.

Outputs (saved for the C port):
    ae_weights.npz  : W1,b1,W2,b2 + NOMINAL, SCALE, THRESHOLD
"""

import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # deterministic numerics
import tensorflow as tf
from tensorflow import keras
from data_loader import load_csv, make_windows

W       = 10        # window length  (input size)
HIDDEN  = 5         # bottleneck size
NOMINAL = 1.0       # expected gap (s)
EPOCHS  = 40
BATCH   = 256
SEED    = 0

tf.random.set_seed(SEED)
np.random.seed(SEED)


def normalize(windows, scale):
    """Center on NOMINAL and scale so the net sees deviations."""
    return (windows - NOMINAL) / scale


def build_model():
    """The tiny dense autoencoder. No bias-free tricks; plain Dense layers."""
    m = keras.Sequential([
        keras.layers.Input(shape=(W,)),
        keras.layers.Dense(HIDDEN, activation="relu", name="enc"),
        keras.layers.Dense(W, activation="linear", name="dec"),
    ])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return m


def recon_error(model, X):
    """Per-window mean-squared reconstruction error."""
    R = model.predict(X, verbose=0)
    return ((R - X) ** 2).mean(axis=1)


if __name__ == "__main__":
    # ---- Load clean training + validation data ----------------------
    train = load_csv("dataset/train_normal.csv")
    val   = load_csv("dataset/val_normal.csv")

    # Scale = std of clean training deviations from nominal.
    SCALE = float(np.std(train["gaps"] - NOMINAL))
    print(f"Normalization: NOMINAL={NOMINAL}, SCALE={SCALE:.6e}")

    # ---- Build windows ----------------------------------------------
    Xtr_raw, _ = make_windows(train["gaps"], W)
    Xva_raw, _ = make_windows(val["gaps"],   W)
    Xtr = normalize(Xtr_raw, SCALE)
    Xva = normalize(Xva_raw, SCALE)
    print(f"train windows: {Xtr.shape}   val windows: {Xva.shape}")

    # ---- Train ------------------------------------------------------
    model = build_model()
    hist = model.fit(
        Xtr, Xtr,
        validation_data=(Xva, Xva),
        epochs=EPOCHS, batch_size=BATCH,
        verbose=0,
    )
    print(f"final train loss: {hist.history['loss'][-1]:.6e}")
    print(f"final val   loss: {hist.history['val_loss'][-1]:.6e}")

    # ---- Set threshold from clean validation errors -----------------
    err_va = recon_error(model, Xva)
    # Threshold = mean + k*std of clean reconstruction error.
    k = 6.0
    THRESHOLD = float(err_va.mean() + k * err_va.std())
    fp = int((err_va > THRESHOLD).sum())
    print(f"clean val recon error: mean={err_va.mean():.3e} "
          f"std={err_va.std():.3e}")
    print(f"threshold (mean+{k:.0f}std): {THRESHOLD:.3e}")
    print(f"false positives on clean val: {fp} / {len(err_va)}")

    # ---- Export weights for the C port ------------------------------
    enc = model.get_layer("enc")
    dec = model.get_layer("dec")
    W1, b1 = enc.get_weights()
    W2, b2 = dec.get_weights()
    np.savez("ae_weights.npz",
             W1=W1, b1=b1, W2=W2, b2=b2,
             NOMINAL=NOMINAL, SCALE=SCALE, THRESHOLD=THRESHOLD, W=W, HIDDEN=HIDDEN)
    print("saved ae_weights.npz")
