import sys
from os.path import dirname, abspath
import re
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from statistics import stdev, mean
from collections import Counter
from src.utils import Utils, is_number
from src.globals import ENT_MASK
from src.DataStructures import SentTrain, SentValid


# Global paths.
CURRENT_DIR = dirname(abspath(__file__))
MAIN_DIR = dirname(CURRENT_DIR)
DATA_DIR = f"{MAIN_DIR}/data"


def pad_sequence(seq, pad_id, pad_len):
    return seq[:pad_len] + [pad_id]*(pad_len-len(seq))


class DataManager:
    def __init__(self, process_data):
        self.train_file = f"{DATA_DIR}/train/data_masked_relabelled.json"
        self.train_processed_file = \
            f"{DATA_DIR}/train/data_masked_relabelled_processed.p"
        self.valid_file = f"{DATA_DIR}/validation/test_relabelled.txt"
        self.valid_processed_file = \
            f"{DATA_DIR}/validation/test_relabelled_processed.p"
        self.vocab_file = f"{DATA_DIR}/train/vocabulary.json"
        self.num_token = "[num]"
        self.stemmer = PorterStemmer()

        if process_data:
            self.process_train_data()
            self.process_valid_data()

    def load_raw_train_data(self):
        return [SentTrain(x) for x in Utils.read_json(self.train_file)]

    def load_raw_valid_data(self):
        return [SentValid(x) for x in Utils.read_txt(self.valid_file)]

    def stem_tokenize_sentences(self, data):
        for x in data:
            x.text_tokens = [
                ENT_MASK if t == "ENT_MASK"  # ENT_MASK = "[ENT]" has to be replaced with the string "ENT_MASK" before tokenization (otherwise it gets tokenized) and then re-replaced with the original mask value "[ENT]".
                else self.num_token if is_number(t)
                else self.stemmer.stem(t.lower())
                for t in word_tokenize(x.text.replace(ENT_MASK, "ENT_MASK"))
                ]

    def build_vocab(self, data, min_df=1):
        counter = Counter([
            t for x in data for t in set(x.text_tokens)
            if t not in [ENT_MASK, self.num_token]  # These 2 will be added to the vocabulary regardless their count.
            ])
        unique_tokens = set([t for t, cnt in counter.items() if cnt >= min_df])
        self.vocab = {
            "[pad]": 0,
            "[unk]": 1,
            ENT_MASK: 2,
            self.num_token: 3,
            **{t: i+4 for i, t in enumerate(sorted(unique_tokens))}
        }
        Utils.write_json(self.vocab, self.vocab_file)

    def convert_tokens_to_ids(self, data):
        unk_id = self.vocab["[unk]"]
        for x in data:
            x.tokenids = [self.vocab.get(t, unk_id) for t in x.text_tokens]

    def get_lenght_stats(self, data):
        lenghts = [len(x.text_tokens) for x in data]
        return int(mean(lenghts)), int(stdev(lenghts))

    def pad_sentences(self, data, pad_len):
        pad_id = self.vocab['[pad]']
        for x in data:
            x.tokenids_pad = pad_sequence(x.tokenids, pad_id, pad_len)
            x.pad_mask = [0 if x == pad_id else 1 for x in x.tokenids_pad]

    def process_train_data(self):
        print("\nPre-Processing train data...")
        data = self.load_raw_train_data()
        self.stem_tokenize_sentences(data)
        self.build_vocab(data, min_df=10)
        self.convert_tokens_to_ids(data)
        avg_len, std_len = self.get_lenght_stats(data)
        self.max_seq_len = avg_len+std_len
        self.pad_sentences(data, pad_len=self.max_seq_len)
        Utils.write_pickle(data, self.train_processed_file)

    def process_valid_data(self):
        print("Pre-Processing valid data...")
        data = self.load_raw_valid_data()
        self.stem_tokenize_sentences(data)
        self.convert_tokens_to_ids(data)
        self.pad_sentences(data, pad_len=self.max_seq_len)
        Utils.write_pickle(data, self.valid_processed_file)

    def load_processed_train_data(self):
        return Utils.read_pickle(self.train_processed_file), \
            Utils.read_json(self.vocab_file)

    def load_processed_valid_data(self):
        return Utils.read_pickle(self.valid_processed_file), \
            Utils.read_json(self.vocab_file)
