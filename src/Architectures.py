import torch
import torch.nn as nn
from torch.autograd import Variable
from math import sin, cos, sqrt


class PositionalEncoder(nn.Module):
    '''
    Compute the sinusoidal positional embeddings which Google is on about
    in da paper.
    '''
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model

        # Create constant 'pe' matrix with values dependant on pos and i.
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i+1] = cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        # Make pos embeddings relatively smaller.
        pe /= sqrt(self.d_model)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def add_pos_embeddings(self, x):
        # Add CONSTANT pos embedding to word embeddings.
        x = x + Variable(self.pe, requires_grad=False).to(x.device.type)
        return x


class SoftmaxMasked(nn.Module):
    '''
    Compute softmax, but only considering the elements that have mask = 1.
    Returns 0 for all the elements that have mask = 0.
    Elements with mask=0 ARE NOT considered when the exponantials are summed
    in the denominators.
    '''
    def __init__(self):
        super().__init__()

    def outer_product(self, x):
        return torch.bmm(x.float().unsqueeze(2), x.float().unsqueeze(1))

    def repeat_mask_for_n_heads(self, matrix, mask):
        '''
        Repeat each mask sequence as many times as the number of heads in the
        self-attention module.
        '''
        bh = matrix.size(0)  # batch*heads (= multi-head batch_size).
        b, t = mask.size()  # "mask" original batch size, and sequence size.
        h = bh//b  # Number of heads.
        return mask.unsqueeze(1).repeat(1, h, 1).view(bh, t)

    def exp_numerator(self, matrix, mask):
        # Mask dimensions are (batch, seq) -> make it into (batch, seq, seq).
        # E.g. mask = [1,1,0] -> mask_matrix=[[1,1,0],[1,1,0],[0,0,0]]
        mask_matrix = self.outer_product(mask)
        exp_matrix = torch.exp(matrix)
        return torch.mul(exp_matrix, mask_matrix)

    def sumexp_denominator(self, exp_matrix):
        device = exp_matrix.device.type
        denom = torch.sum(exp_matrix, dim=-1)
        # Add small value to avoid division by 0 when the row is all 0s.
        epsilon = torch.tensor([[1e-4]]).repeat(denom.shape).to(device)
        denom = torch.add(denom, epsilon)
        denom = denom.unsqueeze(-1).repeat(1, 1, exp_matrix.size(-1))
        return denom

    def forward(self, matrix, mask):
        mask = self.repeat_mask_for_n_heads(matrix, mask)
        exp_matrix_masked = self.exp_numerator(matrix, mask)
        sumexp_masked = self.sumexp_denominator(exp_matrix_masked)
        softmax_masked = torch.div(exp_matrix_masked, sumexp_masked)
        return softmax_masked


class SelfAttention_sinlgeHead(nn.Module):
    def __init__(self, k, n_heads=1):
        super().__init__()
        self.k = k
        self.proj_to_Q = nn.Linear(k, k, bias=False)
        self.proj_to_K = nn.Linear(k, k, bias=False)
        self.proj_to_V = nn.Linear(k, k, bias=False)
        self.softmax_masked = SoftmaxMasked()

    def forward(self, inp, pad_mask):
        Q = self.proj_to_Q(inp)  # Query
        K = self.proj_to_K(inp)  # Key
        V = self.proj_to_V(inp)  # Value

        alpha = torch.bmm(Q, K.transpose(1, 2))
        alpha /= (self.k**(1/2))  # Scale alpha by dimension of the inputs.
        W = self.softmax_masked(alpha, pad_mask)

        return torch.bmm(W, V)


class MultiHeadSelfAttention(nn.Module):
    '''
    Multi-head self attention, with pad_masked propagation.
    '''
    def __init__(self, input_size, n_heads=1):
        super().__init__()
        self.k = input_size
        self.h = n_heads
        self.proj_to_Qs = nn.Linear(self.k, self.k * self.h, bias=False)
        self.proj_to_Ks = nn.Linear(self.k, self.k * self.h, bias=False)
        self.proj_to_Vs = nn.Linear(self.k, self.k * self.h, bias=False)
        self.unifyheads = nn.Linear(self.k * self.h, self.k)
        self.softmax_masked = SoftmaxMasked()

    def forward(self, inp, pad_mask):
        # Get dimensions (b=batch, t=sequence, k=input-embedding, h=heads).
        b, t, k = inp.size()
        h = self.h
        # Project input embedding "inp" into the "h" SEPARATE heads of
        # Queries, Keys and Values.
        Qs = self.proj_to_Qs(inp).view(b, t, h, k)  # Queries
        Ks = self.proj_to_Ks(inp).view(b, t, h, k)  # Keys
        Vs = self.proj_to_Vs(inp).view(b, t, h, k)  # Values

        # Fold heads into the batch dimension.
        Qs = Qs.transpose(1, 2).contiguous().view(b * h, t, k)
        Ks = Ks.transpose(1, 2).contiguous().view(b * h, t, k)
        Vs = Vs.transpose(1, 2).contiguous().view(b * h, t, k)

        alpha = torch.bmm(Qs, Ks.transpose(1, 2))  # alpha dims = (b*h, t, t).
        alpha /= (self.k**(1/2))  # Scale alpha by dimensionality of inputs.
        W = self.softmax_masked(alpha, pad_mask)
        out = torch.bmm(W, Vs).view(b, h, t, k)
        out = out.transpose(1, 2).contiguous().view(b, t, h * k)

        return self.unifyheads(out)


class TransformerBlock(nn.Module):
    def __init__(self, k, heads):
        super().__init__()

        self.attention = MultiHeadSelfAttention(input_size=k, n_heads=heads)
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)
        self.ff = nn.Sequential(
            nn.Linear(k, 4 * k),
            nn.SELU(),
            nn.Linear(4 * k, k))

    def forward(self, x, pad_mask):
        attended = self.attention(x, pad_mask)
        x = self.norm1(attended + x)
        fedforward = self.ff(x)
        return self.norm2(fedforward + x)
