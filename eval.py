from run import load_model, test_with_batch


ids_path = "dataset/test/ids.txt"
calls_path = "dataset/test/calls.txt"


def read_test(calls_path, ids_path):
    with open(calls_path) as f:
        calls_list = [l.strip().split() for l in f.readlines()]
    with open(ids_path) as f:
        ids_list = [l.strip() for l in f.readlines()]

    return dict(zip(ids_list, calls_list))


if __name__ == '__main__':
    model = load_model("lstm_19_07_23")
    print("READING DATA")
    test_dict = read_test()
    result = test_with_batch(model, test_dict)
    with open("dataset/test/result.txt", "w") as f:
        for k, v in result.items():
            s = "{} {}".format(k, v)
            print(s)
            f.write(s+"\n")




