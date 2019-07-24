from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout, initializers
from keras.layers import LSTM
from keras.optimizers import Adam


def build_lstm_model(compile=True):
    from data import MAX_SEQUENCE_LEN, ENCODER
    max_features = len(ENCODER) + 1
    max_len = MAX_SEQUENCE_LEN
    embedding_size = 128

    model = Sequential([
        Embedding(max_features, embedding_size, input_length=max_len, mask_zero=True),
        LSTM(64, activation='sigmoid', recurrent_activation='hard_sigmoid',
             kernel_initializer=initializers.lecun_uniform(seed=None)),
        Dropout(0.5),
        Dense(32, activation='sigmoid'),
        Dropout(0.2),
        Dense(1, activation='sigmoid'),
    ])

    if compile:
        model.compile(loss='binary_crossentropy',
                      # optimizer=RMSprop(),
                      optimizer=Adam(lr=0.0005),
                      metrics=['accuracy'])
        model.summary()

    model.name = "lstm"
    return model


def build_lstm_model_preproc(compile=True):
    from data import MAX_SEQUENCE_LEN, ENCODER
    from config import REDUCED_SEQUENCE_LEN

    max_features = len(ENCODER) + 1
    max_len = MAX_SEQUENCE_LEN
    embedding_size = 128

    model = Sequential([
        Embedding(max_features, embedding_size, input_length=max_len, mask_zero=True),
        LSTM(64, activation='sigmoid', recurrent_activation='hard_sigmoid',
             kernel_initializer=initializers.lecun_uniform(seed=None)),
        Dropout(0.5),
        Dense(32, activation='sigmoid'),
        Dropout(0.2),
        Dense(1, activation='sigmoid'),
    ])

    if compile:
        model.compile(loss='binary_crossentropy',
                      # optimizer=RMSprop(),
                      optimizer=Adam(lr=0.0005),
                      metrics=['accuracy'])
        model.summary()

    model.name = "lstm"
    return model