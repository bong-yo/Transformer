from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import torch
from src.ModelHelpers import create_rndm_batches, convert_to_torch_tensors


class Validator:
    def __init__(self, device):
        self.device = device

    def loss_batch_valid(self, batch):
        sents, pad_masks, labels = \
                convert_to_torch_tensors(zip(*batch), self.device)
        logits = self.model(sents, pad_masks)
        return self.criterion(logits, labels), logits, labels

    def run_validation(self, model, sentences):
        self.model = model
        self.model.eval()
        loss = 0
        pred_labels, true_labels = [], []
        batches = create_rndm_batches(sentences, 1000)

        for batch in tqdm(batches, desc="validation"):
            with torch.no_grad():
                loss_batch, logits, true_labels_batch = \
                    self.loss_batch_valid(batch)
                true_labels.extend(true_labels_batch.tolist())
                pred_labels.extend(logits.argmax(-1).tolist())
                loss += loss_batch.item()
        loss /= len(batches)
        prec, rec, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average="macro"
            )
        return loss, prec, rec, f1
