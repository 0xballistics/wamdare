import json
import random
import os
from keras import preprocessing
import numpy as np
from config import ENCODER_SAVE_PATH, MAX_SEQUENCE_LEN, DATASET_SAVE_PATH, MALWARE_PATH, BENIGN_PATH, FORCE_READ_DATA

ENCODER = {'null': 0}
DECODER = {0: 'null'}


def read_f(path):
    benign = []
    with open(path) as f:
        for line in f:
            calls = line.strip().split()
            benign.append(calls)
    return benign


def create_encoder_decoder(benign_list, malware_list):
    global ENCODER
    global DECODER
    uniq = set()
    for sample in benign_list + malware_list:
        for call in sample:
            uniq.add(call)
    for i, elem in enumerate(uniq, 1):
        ENCODER[elem] = i
        DECODER[i] = elem


def encode_data(calls_list):
    enc = []
    for calls in calls_list:
        calls = calls[:MAX_SEQUENCE_LEN]
        l = [ENCODER.get(c, 0) for c in calls]
        enc.append(l)

    return enc


def decode_data(arr2d):
    dec = []
    for row in arr2d:
        l = []
        for c in row:
            s = DECODER.get(int(c))
            if s == 'null':
                break
            else:
                l.append(s)
        dec.append(l)
    return dec


def split_data(data, limit=None, test_ratio=0.1, val_ratio=0.1):
    if not limit:
        limit = len(data)
    elif limit < len(data):
        data = data[:limit]

    test_count = int(limit * test_ratio)
    val_count = int(limit * val_ratio)
    train_count = int(limit * (1.0 - val_ratio - test_ratio))
    random.shuffle(data)
    test_end = int(len(data) * test_ratio)
    val_end = test_end + int(len(data) * val_ratio)

    print("test: {}:{} val: {}:{} train: {}:{}".format(0, test_end, test_end, val_end, val_end, len(data)))
    test = data[0:test_end]
    val = data[test_end:val_end]
    train = data[val_end:]

    while len(test) < test_count:
        test = test + data[0:test_end]
    while len(val) < val_count:
        val = val + data[test_end:val_end]
    while len(train) < train_count:
        train = train + data[val_end:]

    return train, test, val


def convert_to_numpy(data, padding_type='post'):
    padded = preprocessing.sequence.pad_sequences(data, maxlen=MAX_SEQUENCE_LEN, dtype='int32',
                                                  padding=padding_type, truncating=padding_type, value=0.0)
    return padded


def label_data(benign, malware):
    # merge
    X = np.concatenate((benign, malware), axis=0)
    # labels
    Y = np.concatenate((np.ones(benign.shape[0], dtype=int), np.zeros(malware.shape[0], dtype=int)))
    return X, Y


def prepare_data(data, limit=None, test_ratio=0.1, val_ratio=0.1, padding_type='post'):
    encoded = encode_data(data)
    train, test, val = split_data(encoded, limit, test_ratio, val_ratio)
    trainX = convert_to_numpy(train, padding_type)
    testX = convert_to_numpy(test, padding_type)
    valX = convert_to_numpy(val, padding_type)

    return trainX, testX, valX


def generate_dataset():
    benign_raw = read_f(BENIGN_PATH)
    malware_raw = read_f(MALWARE_PATH)

    create_encoder_decoder(benign_raw, malware_raw)

    benign = prepare_data(benign_raw)
    malware = prepare_data(malware_raw)

    X_train, Y_train = label_data(benign[0], malware[0])
    X_test, Y_test = label_data(benign[1], malware[1])
    X_val, Y_val = label_data(benign[2], malware[2])

    return X_train, Y_train, X_test, Y_test, X_val, Y_val


def print_data_examples(X, y):
    positive = X[y.astype(bool), :]
    negative = X[np.invert(y.astype(bool)), :]

    print("POSITIVE EXAMPLES (out of {}):".format(positive.shape[0]))

    for c in range(8):
        idx = random.randint(0, positive.shape[0]-1)
        sample_raw = positive[idx]
        calls = decode_data([sample_raw])[0][:10]
        print(calls)

    print("NEGATIVE EXAMPLES (out of {}):".format(negative.shape[0]))

    for c in range(8):
        idx = random.randint(0, negative.shape[0] - 1)
        sample_raw = negative[idx]
        calls = decode_data([sample_raw])[0][:10]
        print(calls)


def load_encoder(path):
    global ENCODER
    global DECODER
    with open(path) as f:
        encoder = json.load(f)

    ENCODER = encoder
    DECODER = dict([(v, k) for k, v in encoder.items()])

    return ENCODER, DECODER


def get_data():
    if os.path.exists(DATASET_SAVE_PATH) and not FORCE_READ_DATA:
        load_encoder(ENCODER_SAVE_PATH)
        filez = np.load(DATASET_SAVE_PATH)
        return filez['x_train'], filez['y_train'], filez['x_test'], filez['y_test'], filez['x_val'], filez['y_val']
    else:
        X_train, Y_train, X_test, Y_test, X_val, Y_val = generate_dataset()

        with open(ENCODER_SAVE_PATH, 'w') as f:
            json.dump(ENCODER, f)

        with open(DATASET_SAVE_PATH, 'wb') as f:
            np.savez_compressed(f, x_train=X_train, y_train=Y_train, x_test=X_test, y_test=Y_test, x_val=X_val,
                                y_val=Y_val)
        return X_train, Y_train, X_test, Y_test, X_val, Y_val
