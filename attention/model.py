import torch
import torch.nn as nn

class SimpleAttention(torch.nn.Module):
    """Simple attention mechanism"""

    def __init__(self, dim=256, seq_len=30):
        super().__init__()
        A = torch.empty(dim, seq_len)
        torch.nn.init.xavier_normal_(A)
        self.A = torch.nn.Parameter(A)

    def forward(self, H):
        """input x is the lstm output, its shape is (batch_size, seq_len, lstm_hidden)
        or (batch_size, seq_len, lstm_hidden * 2) if bidirectional"""

        X = torch.nn.functional.softmax(torch.matmul(H, self.A), dim=-1)
        out = torch.matmul(X, H)
        return out

class RNNEncoder(torch.nn.Module):
    def __init__(self,
                 num_embed=4,
                 embed_dim=32,
                 embed_dropout=0,
                 lstm_layers=1,
                 lstm_hidden=128,
                 lstm_dropout=0.5,
                 bidirectional=True,
                 use_attention=False,
                 seq_len=30, #written for sequences with length 30
                 is_final_output=False):
        super().__init__()
        self.src_embed = torch.nn.Embedding(num_embed, embed_dim)
        self.embed_dropout = torch.nn.Dropout(embed_dropout)
        self.lstm = torch.nn.LSTM(input_size=embed_dim,
                                  hidden_size=lstm_hidden,
                                  num_layers=lstm_layers,
                                  batch_first=True,
                                  dropout=lstm_dropout,
                                  bidirectional=bidirectional)

        if bidirectional:
            lstm_hidden *= 2

        if use_attention:
            self.attention_layer = SimpleAttention(
                dim=lstm_hidden, seq_len=seq_len)

        self.is_final_output = is_final_output
        if is_final_output:
            self.out = torch.nn.Linear(lstm_hidden, 3)

    def forward(self, x):
        x = self.src_embed(x)
        x = self.embed_dropout(x)
        x, _ = self.lstm(x)

        if hasattr(self, 'attention_layer'):
            x = self.attention_layer(x)

        if self.is_final_output:
            x = self.out(x)
        return x

class RNAStructNet(torch.nn.Module):
    def __init__(self,
                 encoder='RNN',
                 decoder=None,
                 **encoder_kwargs):
        super().__init__()
        if 'is_final_output' in encoder_kwargs and encoder_kwargs['is_final_output'] and decoder is not None:
            raise ValueError(
                'With is_final_out=True in encoder, decoder must be set to None!')

        if encoder == 'RNN':
            self.encoder = RNNEncoder(**encoder_kwargs)
        elif encoder == 'self_attention':
            self.encoder = SelfAttentionEncoder(**encoder_kwargs)
        elif encoder == 'GCN':
            self.encoder = GCNEncoder(**encoder_kwargs)
        else:
            raise ValueError('Encoder must be one of RNN, self_attention, GCN')

        self.encoder_name = encoder
        self.decoder_name = decoder

    def forward(self, x, adj=None):
        if self.encoder_name == 'GCN':
            assert adj is not None, 'GCN encoder takes two arguments, but adjacency matrix is not provided.'
            x = self.encoder(x, adj)
        else:
            x = self.encoder(x)

        if self.decoder_name == 'symmetric_matrix':
            return torch.matmul(x, x.transpose(1, 2))
        else:
            return x

    def __repr__(self):
        return self.__class__.__name__ + '(encoder: ' + encoder + ', decoder: ' + decoder + ')'
