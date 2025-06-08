import joblib
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import (
    Input,
    Dense,
    Conv1D,
    Layer,
    Flatten,
    Lambda,
    Reshape,
    Concatenate,
    GlobalAveragePooling1D,
    MaxPooling1D,
    Dropout,
    BatchNormalization,
    GaussianNoise,
)
from ..preprocess.augmentation import augment_data
from tensorflow.keras.utils import register_keras_serializable
from .common import load_parquet, load_model, load_scaler
from . import constants as c
from tensorflow.keras.callbacks import EarlyStopping


def build_word_autoencoder(input_dim, encoding_dim=6):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation="relu")(input_layer)
    encoded = Dense(32, activation="relu")(encoded)
    encoded = Dense(16, activation="relu")(encoded)
    encoded = Dense(8, activation="relu")(encoded)
    bottleneck = Dense(encoding_dim, activation="relu", name="bottleneck")(encoded)
    decoded = Dense(8, activation="relu")(bottleneck)
    decoded = Dense(16, activation="relu")(decoded)
    decoded = Dense(32, activation="relu")(decoded)
    decoded = Dense(64, activation="relu")(decoded)
    decoded = Dense(input_dim, activation="sigmoid", name="reconstruction")(decoded)

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder, input_layer, bottleneck


def build_global_autoencoder(
    input_dim,
    encoding_dim=8,
    intermediate_dims=(96, 64, 32, 16),
    denoising=False,
    noise_std=0.05,
    lr=1e-3,
):
    input_layer = Input(shape=(input_dim,))
    if denoising:
        input_layer = GaussianNoise(noise_std)(input_layer)
    x = input_layer
    for d in intermediate_dims:
        x = Dense(d, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
    bottleneck = Dense(encoding_dim, activation="relu", name="bottleneck")(x)
    x_dec = bottleneck
    for i, d in enumerate(reversed(intermediate_dims)):
        x_dec = Dense(d, activation="relu")(x_dec)
    decoded = Dense(input_dim, activation="sigmoid", name="reconstruction")(x_dec)

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=lr), loss="mse")
    return autoencoder


@register_keras_serializable(package="custom_vae")
def sampling(args):
    z_mean, z_log_var = args
    eps = K.random_normal(shape=(K.shape(z_mean)[0], K.shape(z_mean)[1]))
    return z_mean + K.exp(0.5 * z_log_var) * eps

@register_keras_serializable(package="custom_vae")
def vae_loss_fn(args):
    x, x_decoded, z_mean, z_log_var = args
    # reconstruction loss per sample
    recon = K.sum(K.square(x - x_decoded), axis=-1)
    # KL divergence per sample
    kl = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    # combine and take batch mean
    return K.mean(recon + kl)

@register_keras_serializable(package="custom_vae")
class VAELossLayer(Layer):
    def call(self, inputs):
        x_true, x_pred, z_mean, z_log_var = inputs
        # per-sample reconstruction loss
        recon = K.sum(K.square(x_true - x_pred), axis=-1)
        # per-sample KL divergence
        kl = -0.5 * K.sum(1 + z_log_var
                          - K.square(z_mean)
                          - K.exp(z_log_var),
                          axis=-1)
        # register the batch-mean total loss
        self.add_loss(K.mean(recon + kl))
        # pass predictions onward unchanged
        return x_pred
# 0.5639 -> 256, 128, 64, 32, 16, 8 -> 6
# 0.4756 -> 256, 128, 64, 32, 16 -> 14
# 0.4433 -> 512, 256, 128, 64, 32 -> 16
def build_variational_autoencoder(
    input_dim,
    latent_dim=10,
    intermediate_dims=(512, 256, 128, 64, 32),
    denoising=False,
    noise_std=0.05,
    lr=1e-3,
):
    # 1) ENCODER
    x_in = Input(shape=(input_dim,), name="encoder_input")
    x = x_in
    if denoising:
        x = GaussianNoise(noise_std, name="gaussian_noise")(x)

    for i, d in enumerate(intermediate_dims):
        x = Dense(d, activation="relu", name=f"enc_dense_{i}")(x)

    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)

    z = Lambda(sampling, name="z")([z_mean, z_log_var])
    encoder = Model(x_in, [z_mean, z_log_var, z], name="encoder")

    # 2) DECODER
    z_in = Input(shape=(latent_dim,), name="decoder_input")
    x_dec = z_in
    for i, d in enumerate(reversed(intermediate_dims)):
        x_dec = Dense(d, activation="relu", name=f"dec_dense_{i}")(x_dec)
    x_out = Dense(input_dim, activation="sigmoid", name="decoder_output")(x_dec)
    decoder = Model(z_in, x_out, name="decoder")

    # 3) CONNECT: full VAE
    z_sampled = encoder(x_in)[2]
    x_decoded = decoder(z_sampled)

    # 4) INJECT LOSS via custom layer
    x_with_loss = VAELossLayer(name="vae_loss")(
        [x_in, x_decoded, z_mean, z_log_var]
    )

    vae = Model(x_in, x_with_loss, name="vae")
    vae.compile(optimizer=Adam(learning_rate=lr))
    return vae, encoder, decoder



def build_multi_conv_autoencoder(non_seq_dim=9, encoding_dim=5):
    # Inputs
    input_nonseq = Input(shape=(non_seq_dim,), name="nonseq_input")
    input_chaincode = Input(shape=(8,), name="chaincode_input")
    input_hod = Input(shape=(9,), name="hod_input")
    input_viz = Input(shape=(3,), name="viz_input")

    # Conv branches
    x_chain = Reshape((8, 1))(input_chaincode)
    conv_chain = Conv1D(16, 3, activation="relu", padding="same")(x_chain)
    conv_chain = MaxPooling1D(pool_size=2)(conv_chain)
    conv_chain = Conv1D(4, 3, activation="relu", padding="same")(conv_chain)
    conv_chain = Conv1D(2, 3, activation="relu", padding="same")(conv_chain)
    conv_chain = Flatten()(conv_chain)

    x_hod = Reshape((9, 1))(input_hod)
    conv_hod = Conv1D(16, 3, activation="relu", padding="same")(x_hod)
    conv_hod = MaxPooling1D(pool_size=2)(conv_hod)
    conv_hod = Conv1D(4, 3, activation="relu", padding="same")(conv_hod)
    conv_hod = Conv1D(2, 3, activation="relu", padding="same")(conv_hod)
    conv_hod = Flatten()(conv_hod)

    x_viz = Reshape((3, 1))(input_viz)
    conv_viz = Conv1D(8, 2, activation="relu", padding="same")(x_viz)
    conv_viz = Conv1D(2, 2, activation="relu", padding="same")(conv_viz)
    conv_viz = Flatten()(conv_viz)

    # Dense branch for non-sequential
    dense_nonseq = Dense(16, activation="relu")(input_nonseq)

    # Merge all
    merged = Concatenate()([dense_nonseq, conv_chain, conv_hod, conv_viz])

    # Encoder
    encoded = Dense(48, activation="relu")(merged)
    encoded = Dense(32, activation="relu")(encoded)
    encoded = Dense(16, activation="relu")(encoded)
    bottleneck = Dense(encoding_dim, activation="relu", name="encoded")(encoded)

    # Decoder
    decoder_hidden = Dense(16, activation="relu")(bottleneck)
    decoder_hidden = Dense(32, activation="relu")(decoder_hidden)
    decoder_hidden = Dense(48, activation="relu")(decoder_hidden)

    # Reconstruction branches
    decoded_nonseq = Dense(non_seq_dim, activation="sigmoid", name="decoded_nonseq")(
        decoder_hidden
    )
    decoded_chain = Dense(8, activation="sigmoid", name="decoded_chaincode")(
        decoder_hidden
    )
    decoded_hod = Dense(9, activation="sigmoid", name="decoded_hod")(decoder_hidden)
    decoded_viz = Dense(3, activation="sigmoid", name="decoded_viz")(decoder_hidden)

    # Full model
    autoencoder = Model(
        inputs=[input_nonseq, input_chaincode, input_hod, input_viz],
        outputs=[decoded_nonseq, decoded_chain, decoded_hod, decoded_viz],
    )
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder


def train_writer_autoencoder(writer_id):

    print(f"Training writer {writer_id}...")
    os.makedirs(os.path.join(c.MODEL_FOLDER, c.AUTOENCODERS, c.TRAIN), exist_ok=True)

    train = load_parquet(f"{c.FEATURES}/{c.TRAIN}/{writer_id}")
    val = load_parquet(f"{c.FEATURES}/{c.VAL_OWN}/{writer_id}")

    if train is None or val is None:
        print(f"Feature file not found")
        return

    train = augment_data(train)
    train = augment_data(train)
    train = train.drop(columns=["sample"])
    val = val.drop(columns=["sample"])

    # scaler = load_scaler(
    #     f"{c.MODEL_FOLDER}/{c.AUTOENCODERS}/{c.PRETRAINED}/pretrained_scaler"
    # )
    scaler = MinMaxScaler()
    autoencoder = build_global_autoencoder(train.shape[1])
    autoencoder.load_weights(
        f"{c.MODEL_FOLDER}/{c.AUTOENCODERS}/{c.PRETRAINED}/pretrained.weights.h5"
    )

    train = scaler.fit_transform(train)
    val = scaler.transform(val)
    # scaler_filename = os.path.join(model_folder, sub_dir, f"{writer_id}_scaler.pkl")
    # joblib.dump(scaler, scaler_filename)

    es = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

    autoencoder.fit(
        train,
        train,
        epochs=400,
        batch_size=16,
        shuffle=True,
        verbose=1,
        validation_data=(val, val),
        callbacks=[es],
    )

    autoencoder.save_weights(
        f"{c.MODEL_FOLDER}/{c.AUTOENCODERS}/{c.TRAIN}/{writer_id}_autoencoder.weights.h5"
    )
    joblib.dump(
        scaler, f"{c.MODEL_FOLDER}/{c.AUTOENCODERS}/{c.TRAIN}/{writer_id}_scaler.pkl"
    )


def train_writer_conv_autoencoder(writer_id):

    print(f"Training writer {writer_id}...")
    os.makedirs(
        os.path.join(c.MODEL_FOLDER, c.CONV_AUTOENCODERS, c.TRAIN), exist_ok=True
    )

    data = load_parquet(f"{c.FEATURES}/{c.TRAIN}/{writer_id}")
    val = load_parquet(f"{c.FEATURES}/{c.VAL_OWN}/{writer_id}")
    if data is None or val is None:
        print(f"Feature file not found: {data}")
        return

    data = data.drop(columns=["sample"])
    val = val.drop(columns=["sample"])

    X_nonseq = data.iloc[:, 0:25].values.astype(np.float32)
    X_chain = data.iloc[:, 25:33].values.astype(np.float32)
    X_hod = data.iloc[:, 33:42].values.astype(np.float32)
    X_viz = data.iloc[:, 42:].values.astype(np.float32)

    val_nonseq = val.iloc[:, 0:25].values.astype(np.float32)
    val_chain = val.iloc[:, 25:33].values.astype(np.float32)
    val_hod = val.iloc[:, 33:42].values.astype(np.float32)
    val_viz = val.iloc[:, 42:].values.astype(np.float32)

    scaler_names = ["nonseq", "chain", "hod", "viz"]
    scalers = [MinMaxScaler() for _ in range(4)]

    X_nonseq = scalers[0].fit_transform(X_nonseq)
    X_chain = scalers[1].fit_transform(X_chain)
    X_hod = scalers[2].fit_transform(X_hod)
    X_viz = scalers[3].fit_transform(X_viz)

    val_nonseq = scalers[0].transform(val_nonseq)
    val_chain = scalers[1].transform(val_chain)
    val_hod = scalers[2].transform(val_hod)
    val_viz = scalers[3].transform(val_viz)

    autoencoder = load_model(
        f"{c.MODEL_FOLDER}/{c.CONV_AUTOENCODERS}/{c.PRETRAINED}/pretrained_autoencoder"
    )

    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    autoencoder.fit(
        [X_nonseq, X_chain, X_hod, X_viz],
        [X_nonseq, X_chain, X_hod, X_viz],
        epochs=400,
        batch_size=16,
        shuffle=True,
        verbose=1,
        validation_data=[
            [val_nonseq, val_chain, val_hod, val_viz],
            [val_nonseq, val_chain, val_hod, val_viz],
        ],
        callbacks=[es],
    )

    autoencoder.save(
        f"{c.MODEL_FOLDER}/{c.CONV_AUTOENCODERS}/{c.TRAIN}/{writer_id}_autoencoder.keras"
    )
    for i, scaler in enumerate(scalers):
        scaler_filename = os.path.join(
            f"{c.MODEL_FOLDER}/{c.CONV_AUTOENCODERS}/{c.TRAIN}/{writer_id}_{scaler_names[i]}_scaler.pkl"
        )
        joblib.dump(scaler, scaler_filename)
