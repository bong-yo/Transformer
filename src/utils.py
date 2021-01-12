import json
import pickle


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


class Utils:
    @staticmethod
    def read_json(filename):
        with open(filename, "r", encoding="utf8") as f:
            return json.load(f)

    @staticmethod
    def write_json(data, filename):
        with open(filename, "w", encoding="utf8") as f:
            json.dump(data, f)

    @staticmethod
    def read_txt(filename):
        with open(filename, "r", encoding="utf8") as f:
            return f.read().splitlines()

    @staticmethod
    def write_txt(data, filename):
        with open(filename, "w", encoding="utf8") as f:
            f.write(data)

    @staticmethod
    def read_pickle(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def write_pickle(data, filename):
        with open(filename, "wb") as f:
            pickle.dump(data, f)
