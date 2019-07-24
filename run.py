import datetime
import os
import config
from keras.engine.saving import model_from_json
from data import load_encoder, encode_data, convert_to_numpy, get_data, print_data_examples
from model import build_lstm_model
from preproc import train_custom_encoder


def save_model(model, text=None):
    save_dir = config.MODEL_SAVE_DIR
    model_raw_name = model.name
    date_suffix = datetime.datetime.now().strftime('_%y_%m_%d')

    model_index = 0
    while True:
        model_index_suffix = "_{}".format(model_index) if model_index else ""
        model_name = "{}/{}{}{}".format(save_dir, model_raw_name, date_suffix, model_index_suffix)
        json_name = model_name + ".json"
        h5_name = model_name + ".h5"
        text_name = model_name + ".txt"
        if os.path.exists(json_name):
            model_index += 1
        else:
            break

    # serialize model to JSON
    model_json = model.to_json()
    with open(json_name, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(h5_name)

    if text:
        with open(text_name, "w") as txt_file:
            txt_file.write(text)


def load_model(model_name):
    save_dir = config.MODEL_SAVE_DIR
    json_file = open('{}/{}.json'.format(save_dir, model_name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("{}/{}.h5".format(save_dir, model_name))

    return loaded_model


def test_model(model, X_test, y_test):
    score = model.evaluate(X_test, y_test, batch_size=50)
    print(score)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return score


def train_test(model, X_train, y_train, X_test, y_test, X_val, y_val):
    print("TRAINING MODEL")
    model.fit(x=X_train, y=y_train, batch_size=config.TRAIN_BATCH_SIZE, epochs=config.EPOCHS, shuffle=True, verbose=1, validation_data=(X_val, y_val))

    print("TESTING MODEL")
    score = test_model(model, X_test, y_test)
    score_text = 'Model: {}\nTest loss: {}\nTest accuracy: {}'.format(model.name, *score)
    print(score_text)
    if config.SAVE_MODEL:
        save_model(model, text=score_text)


def test_with_batch(model, batch_dict: dict, padding_type='post'):
    keys = []
    values = []
    for k,v in batch_dict.items():
        keys.append(k)
        values.append(v)

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    graph = tf.get_default_graph()
    with graph.as_default():
        load_encoder(config.ENCODER_SAVE_PATH)
        data = encode_data(values)
        X = convert_to_numpy(data, padding_type)

        predictions = model.predict(X)
        out = []
        for name, pred in zip(keys, list(predictions)):
            out.append((name, pred[0]))
        return dict(out)


def run_lstm():
    print("LOADING DATA")
    full_set = get_data()
    print_data_examples(full_set[0], full_set[1])
    print("BUILDING MODEL")
    model = build_lstm_model()
    train_test(model, *full_set)


def build_preprocessor():
    print("LOADING DATA")
    X_train, _, X_test, _, _, _= get_data()
    encoder = train_custom_encoder(X_train, X_test)
    save_model(encoder)


def preprocess():
    X_train, y_train, X_test, y_test, X_val, y_val = get_data()
    encoder = load_model(config.AUTOENCODER_NAME)

    X_train_p = encoder.predict(X_train)
    X_test_p = encoder.predict(X_test)
    X_val_p = encoder.predict(X_val)





def run_lstm_with_preprocessor():

    print("LOADING DATA")
    full_set = get_data()
    print_data_examples(full_set[0], full_set[1])
    print("BUILDING MODEL")
    model = build_lstm_model()
    train_test(model, *full_set)


if __name__ == '__main__':
    run_lstm()
