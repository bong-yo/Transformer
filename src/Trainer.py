from tqdm import tqdm
import torch.nn as nn
from copy import deepcopy
from src.Optimizer import Optimizer
from src.ModelHelpers import create_rndm_batches, convert_to_torch_tensors
from src.Validator import Validator


class Trainer(Optimizer, Validator):
    def __init__(self, model, data_train, data_valid, learning_rate, device):
        Optimizer.__init__(self, model, learning_rate)
        Validator.__init__(self, device)
        self.model = model
        self.data_train = data_train
        self.data_valid = data_valid
        self.criterion = nn.CrossEntropyLoss()
        self.device = device
        self.best_loss_valid = None
        self.best_epoch = 0
        self.TRAIN_STATUS_MSG = "Epoch: {}/{} | loss train: {:03.3f} | " + \
            "loss valid: {:03.3f} | prec valid: {:03.2f} | " + \
            "rec valid: {:03.2f} | f1 valid: {:03.2f}."

    def select_best_model(self, loss, epoch):
        if self.best_loss_valid is None:
            self.best_loss_valid = loss
            self.best_epoch = epoch
            self.best_model = deepcopy(self.model)
        elif loss < self.best_loss_valid:
            self.best_loss_valid = loss
            self.best_epoch = epoch
            self.best_model = deepcopy(self.model)

    def loss_batch(self, batch):
        sents, pad_masks, labels = \
            convert_to_torch_tensors(zip(*batch), self.device)
        logits = self.model(sents, pad_masks)
        return self.criterion(logits, labels)

    def train(self, n_epochs, batch_size):
        for epoch in range(n_epochs):
            self.model.train()
            batches = create_rndm_batches(self.data_train, batch_size)
            loss_train = 0

            for batch in tqdm(batches, desc="epoch %d" % (epoch+1)):
                self.model.zero_grad()
                loss = self.loss_batch(batch)
                loss.backward()
                self.optim.step()
                loss_train += loss.item()

            loss_train /= len(batches)
            loss_valid, prec_valid, rec_valid, f1_valid = \
                self.run_validation(self.model, self.data_valid)
            print(self.TRAIN_STATUS_MSG.format(epoch+1, n_epochs, loss_train,
                  loss_valid, prec_valid, rec_valid, f1_valid))
            # Copy model if it's best.
            self.select_best_model(loss_valid, epoch)

        print("\n\n**** Best model is from epoch %d with validation " +
              "loss %.3f**** \n\n" % (self.best_epoch, self.best_loss_valid))
        return self.best_model
