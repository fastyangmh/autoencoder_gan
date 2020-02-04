# import
import torch
import torch.nn as nn

# def


def calculate_dim(in_dim, code_dim, alpha):
    sizes = [in_dim]
    while sizes[-1] > code_dim:
        sizes.append(int(sizes[-1]*alpha))
        if sizes[-1] < code_dim:
            sizes.pop()
            break
    return sizes

# class


class AUTOENCODER(nn.Module):
    def __init__(self, in_dim, condition_dim, code_dim, alpha=0.2, dropout=0.5):
        super(AUTOENCODER, self).__init__()
        enc_layer_sizes = calculate_dim(in_dim+condition_dim, code_dim, alpha)
        dec_layer_sizes = [v for v in enc_layer_sizes[::-1]
                           [:-1]+[in_dim] if v > (code_dim+condition_dim)]
        self.enc_layers = nn.ModuleList([nn.Linear(in_size, out_size) for (
            in_size, out_size) in zip(enc_layer_sizes, enc_layer_sizes[1:])])
        self.last_enc_layer = nn.Linear(enc_layer_sizes[-1], code_dim)
        self.first_dec_layer = nn.Linear(
            code_dim+condition_dim, dec_layer_sizes[0])
        self.dec_layers = nn.ModuleList([nn.Linear(in_size, out_size) for (
            in_size, out_size) in zip(dec_layer_sizes, dec_layer_sizes[1:])])
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, c):
        x = torch.cat((x, c), 1)
        for enc_layer in self.enc_layers:
            x = self.relu(self.dropout(enc_layer(x)))
        x = self.tanh(self.dropout(self.last_enc_layer(x)))
        x = torch.cat((x, c), 1)
        x = self.relu(self.dropout(self.first_dec_layer(x)))
        for dec_layer in self.dec_layers:
            x = self.relu(self.dropout(dec_layer(x)))
        return x

    def encoder(self, x, c):
        x = torch.cat((x, c), 1)
        for enc_layer in self.enc_layers:
            x = self.relu(self.dropout(enc_layer(x)))
        x = self.tanh(self.dropout(self.last_enc_layer(x)))
        return x

    def decoder(self, x, c):
        x = torch.cat((x, c), 1)
        x = self.relu(self.dropout(self.first_dec_layer(x)))
        for dec_layer in self.dec_layers:
            x = self.relu(self.dropout(dec_layer(x)))
        return x
