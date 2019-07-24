from keras.layers import Input, Dense
from keras.models import Model
from config import MAX_SEQUENCE_LEN, REDUCED_SEQUENCE_LEN
from data import get_data
from keras.models import Sequential


def train_custom_encoder(X_train, X_test):

    # this is the size of our encoded representations
    encoding_dim = REDUCED_SEQUENCE_LEN
    # this is our input placeholder
    input_type = Input(shape=(MAX_SEQUENCE_LEN,))

    autoencoder = Sequential([
        Dense(encoding_dim, input_shape=(input_type,), activation='relu'),
        Dense(input_type, activation='sigmoid')
    ])

    input_seq = Input(shape=(input_type,))
    encoder_layer = autoencoder.layers[0]
    encoder = Model(input_seq, encoder_layer(input_seq))

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.summary()

    autoencoder.name = "autoencoder"
    encoder.name = "custom_encoder"

    autoencoder.fit(X_train, X_train,
                    epochs=10,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(X_test, X_test))

    return encoder

