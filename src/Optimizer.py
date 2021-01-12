from torch.optim import Adam


class Optimizer:

    def __init__(self, model, learning_rate):
        # param_optimizer = list(model.named_parameters())
        # no_decay = ['bias', 'gamma', 'beta']
        # # Original parameters were :
        # #self.optimizer_grouped_parameters = {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay_rate': 0.01},
        # # {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],'weight_decay_rate': 0.0},
        self.optim = Adam(model.parameters(), lr=learning_rate)
