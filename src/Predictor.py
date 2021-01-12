from tqdm import tqdm
import torch


class Predictor:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def create_batches(self, data, size):
        n_batches = len(data)//size + int(len(data) % size != 0)
        return [data[i*size: (i+1)*size] for i in range(n_batches)]

    def unzip_batch(self, batch):
        inps, pad_mask = zip(*batch)
        return torch.LongTensor(inps).to(self.device), \
            torch.LongTensor(pad_mask).to(self.device)

    def predict(self, sentences, batch_size=500):
        self.model.eval()
        batches = self.create_batches(sentences, batch_size)
        pred_labels = []
        for batch in tqdm(batches, desc="prediction"):
            inps, pad_mask = self.unzip_batch(batch)
            logits = self.model(inps, pad_mask)
            pred_labels.append(logits.argmax(-1).tolist())
        return [label for batch in pred_labels for label in batch]
