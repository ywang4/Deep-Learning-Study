'''
    A seq2seq model with bidirectional rnn as encoder and
'''

import torch
import torch.nn as nn

class EncoderRNN(nn.Module):
    def __init__(self,hidden_size, num_words, emb_size, num_layers=1, dropout=0):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # word embeddings
        self.embedding = nn.Embedding(num_words, emb_size)








