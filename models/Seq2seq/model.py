'''
    A seq2seq model with bidirectional rnn as encoder and
'''

import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self,hidden_size, embedding, num_layers=1, dropout=0):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # word embeddings
        self.embedding = embedding

        self.gru = nn.GRU(hidden_size, hidden_size, num_layers,
                          dropout=(0 if num_layers ==1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # pad the input_seq in the same length
        padded_input_seq = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)

        outputs, hidden = self.gru(padded_input_seq, hidden)

        # unpack the padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)

        # what is this line doing
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        return outputs, hidden


class AttnLayfer(nn.Module):
    def __init__(self, method, hidden_size):
        super.__init__()












