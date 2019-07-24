import datetime
import os
import config
import json
from run import load_model, test_with_batch

if __name__ == '__main__':
    model = load_model("<model_name>")
    test_with_batch(model, <vboat_calls_dict>)
