import pickle
import os
max_bytes = 2**31 - 1


def pickle_object(obj, save_location):
    with open(save_location, 'wb') as f:
        pickle.dump(obj, f)


def pickle_big_data(data, file_path):
    bytes_out = pickle.dumps(data, protocol=4)
    with open(file_path, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])


def unpickle_big_data(file_path):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except:
        bytes_in = bytearray(0)
        input_size = os.path.getsize(file_path)
        with open(file_path, 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        return pickle.loads(bytes_in)

