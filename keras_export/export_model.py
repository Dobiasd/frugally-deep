import json
import sys

from keras.models import load_model
from keras import backend as K

import numpy as np

def write_text_file(path, text):
    with open(path, "w") as text_file:
        print(text, file=text_file)

def arr_as_arr3(arr):
    assert arr.shape[0] == 1
    depth = len(arr.shape)
    if depth == 2:
        return arr.reshape(1, 1, *arr.shape[1:])
    if depth == 3:
        return arr.reshape(1, *arr.shape[1:])
    if depth == 4:
        return arr.reshape(arr.shape[1:])
    else:
        raise ValueError('invalid number of dimensions')

def show_tensor3(tens):
    return {
        'shape': tens.shape,
        'values': tens.flatten().tolist()
    }

def show_test_data_as_3tensor(arr):
    return show_tensor3(arr_as_arr3(arr))

def gen_test_data(model):
    data_in = list(map(lambda l: np.random.random((1, *l.input_shape[1:])),
        model.input_layers))
    data_out = model.predict(data_in)
    return {
        'inputs': list(map(show_test_data_as_3tensor, data_in)),
        'outputs': list(map(show_test_data_as_3tensor, data_out))
    }

def get_all_weights(model):
    # checks if all layer names are globally unique, recursively
    return [1,2,3]

def main():
    usage = 'usage: [Keras model in HDF5 format] [output path] [test count = 3]'
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print(usage)
        sys.exit(1)
    else:
        in_path = sys.argv[1]
        out_path = sys.argv[2]
        test_count = 3
        if len(sys.argv) == 4:
            test_count = int(sys.argv[3])
        model = load_model(in_path)

        # todo remove
        write_text_file(out_path + '.keras.yml', model.to_yaml()) 

        # todo remove
        write_text_file(out_path + '.keras.json',
            json.dumps(model.to_json(),
            allow_nan=False, indent=2, sort_keys=True))

        json_output = {}

        json_output['architecture'] = json.loads(model.to_json())

        json_output['weights'] = get_all_weights(model)

        json_output['tests'] = [gen_test_data(model) for _ in range(test_count)]

        write_text_file(out_path,
            json.dumps(json_output, allow_nan=False, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
