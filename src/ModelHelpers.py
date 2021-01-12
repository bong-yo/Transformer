from random import shuffle
import torch
import time
import os
import json


def create_rndm_batches(sents, size):
    shuffle(sents)
    n_batches = len(sents)//size + int(len(sents) % size != 0)
    return [sents[i*size: (i+1)*size] for i in range(n_batches)]


def convert_to_torch_tensors(tensor_list, device):
    for tensor in tensor_list:
        yield torch.LongTensor(tensor).to(device)


class ModelLoader:
    '''
    Handles saving and loading of the fine-tuned model.
    '''
    def __init__(self, model, saves_folder, vocabulary={}, label2id={}):
        self.model = model
        self.saves_folder = saves_folder
        self.vocabulary = vocabulary
        self.label2id = label2id

    def save(self):
        print("\nSaving model in %s..." % self.saves_folder)
        start = time.time()
        # Create folder if it doesn't exist already.
        if not os.path.exists(self.saves_folder):
            os.makedirs(self.saves_folder)
        # Save weights.
        filename = f"{self.saves_folder}/model_weights.pt"
        torch.save(self.model.state_dict(), filename)
        # Save hyper-parameters.
        params_dict = {
            "vocab_size": self.model.vocab_size,
            "seq_len": self.model.seq_len,
            "hidden_size": self.model.hidden_size,
            "n_heads": self.model.n_heads,
            "n_classes": self.model.n_classes,
            "transformer_depth": self.model.transformer_depth
            }
        with open(f"{self.saves_folder}/model_parameters.json", 'w') as f:
            json.dump(params_dict, f)
        # Save vocabulary.
        with open(f"{self.saves_folder}/vocabulary.json", 'w') as f:
            json.dump(self.vocabulary, f)
        # Save labels dict.
        with open(f"{self.saves_folder}/label2id.json", 'w') as f:
            json.dump(self.label2id, f)
        print("time : %.1f sec" % (time.time()-start))

    def load(self):
        print("\nLoading model from %s..." % self.saves_folder)
        start = time.time()
        # Load model parameters.
        with open(f"{self.saves_folder}/model_parameters.json", 'r') as f:
            params_dict = json.load(f)
        self.model.vocab_size = params_dict["vocab_size"]
        self.model.seq_len = params_dict["seq_len"]
        self.model.hidden_size = params_dict["hidden_size"]
        self.model.n_heads = params_dict["n_heads"]
        self.model.n_classes = params_dict["n_classes"]
        self.model.transformer_depth = params_dict["transformer_depth"]
        # Initialize architecture based on parameters.
        self.model.init_weights()
        # Load weigths.
        filename = f"{self.saves_folder}/model_weights.pt"
        self.model.load_state_dict(torch.load(filename))
        # Load vocabularies.
        with open(f"{self.saves_folder}/vocabulary.json", 'r') as f:
            vocabulary = json.load(f)
        with open(f"{self.saves_folder}/label2id.json", 'r') as f:
            label2id = json.load(f)
        print("time : %.1f sec" % (time.time()-start))
        return self.model, vocabulary, label2id
