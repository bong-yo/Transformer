import torch
import torch.nn as nn
from src.Architectures import TransformerBlock, PositionalEncoder


class SentenceClassifier(nn.Module):
    def __init__(self, vocab_size=0, hidden_size=0, seq_len=0, n_heads=0, 
                 n_classes=0, trans_depth=0):
        super(SentenceClassifier, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.n_heads = n_heads
        self.n_classes = n_classes
        self.transformer_depth = trans_depth

    def init_weights(self):
        self.pos_encoder = PositionalEncoder(self.hidden_size, self.seq_len)
        self.token_embed = nn.Embedding(self.vocab_size, self.hidden_size)
        self.trans_block = TransformerBlock(self.hidden_size, self.n_heads)
        self.dropout = nn.Dropout(0.3)
        self.proj_out = nn.Linear(self.hidden_size, self.n_classes)

    def forward(self, sentences, pad_mask):
        tok_embeds = self.token_embed(sentences)
        x = self.pos_encoder.add_pos_embeddings(tok_embeds)
        for i in range(self.transformer_depth):
            x = self.trans_block(x, pad_mask)
        x = torch.mean(x, dim=1)  # Get mean of output as sequence encoder.
        x = nn.SELU()(x)
        x = self.dropout(x)
        return self.proj_out(x)
