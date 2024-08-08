import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from scipy.linalg import qr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_orthogonal_matrix(M, N):
    m = torch.randn(N, N)
    return torch.linalg.qr(m, "reduced")[0].t()[:M]  # orthogonal matrix with shape of (M, N)

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.dropout = nn.Dropout(0.01)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, num_layers=num_layers, bidirectional=bidirectional)
        
        for param in self.lstm.named_parameters():
            layer_name, wt = param
            if 'bias' in layer_name:
                torch.nn.init.zeros_(wt)
            else:
                if '_hh_' in layer_name:
                    for idx in range(4):
                        wt = get_orthogonal_matrix(*wt.shape)
                else:
                    torch.nn.init.normal_(wt, mean=0.0, std=1.0) 
        
    def forward(self, input):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        output, hidden = self.lstm(embedded)
        return output, hidden

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size, num_layers=1, bidirectional=False):
        super(BahdanauAttention, self).__init__()
        
        self.bidirectional = bidirectional
        self.num_layers = num_layers

        if bidirectional:
            self.W1 = nn.Linear(hidden_size*2, hidden_size*2)
            self.W2 = nn.Linear(hidden_size*2, hidden_size*2)
            self.V = nn.Linear(hidden_size*2, 1)
        else:
            self.W1 = nn.Linear(hidden_size, hidden_size)
            self.W2 = nn.Linear(hidden_size, hidden_size)
            self.V = nn.Linear(hidden_size, 1)
        
        torch.nn.init.zeros_(self.V.weight)
        torch.nn.init.zeros_(self.V.bias)
        
        for name, param in self.W1.named_parameters():
            if 'weight' in name:
                torch.nn.init.normal_(param)
            else:
                torch.nn.init.zeros_(param)
        
        for name, param in self.W2.named_parameters():
            if 'weight' in name:
                torch.nn.init.normal_(param)
            else:
                torch.nn.init.zeros_(param)

    def forward(self, query, values, mask):
        # Additive attention
        
        # RNN h format: l1_f, l1_b, l2_f, l2_b...
        query = [x[:, -1:, :] for x in query] if not self.bidirectional else \
                [torch.cat((x[:,-1:,:], x[:,-2:-1,:]), dim=-1) \
                    for x in query]
                        # last layer outputs

        scores = self.V(torch.tanh(self.W1(query[0]) + self.W2(values[0])))
        scores = scores.squeeze(2).unsqueeze(1) # [B, M, 1] -> [B, 1, M]

        # Dot-Product Attention: score(s_t, h_i) = s_t^T h_i
        # Query [B, 1, D] * Values [B, D, M] -> Scores [B, 1, M]
        # scores = torch.bmm(query, values.permute(0,2,1))

        # Cosine Similarity: score(s_t, h_i) = cosine_similarity(s_t, h_i)
        # scores = F.cosine_similarity(query, values, dim=2).unsqueeze(1)

        # Mask out invalid positions.
        #Attempting PAD-PAD learning
        #scores.data.masked_fill_(mask.unsqueeze(1) == 0, -float('inf'))

        # Attention weights
        alphas = F.softmax(scores, dim=-1)

        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, values)

        # context shape: [B, 1, D], alphas shape: [B, 1, M]
        return context, alphas


class AttnDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1, bidirectional=False):
        super(AttnDecoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(0.01)
        self.attention = BahdanauAttention(hidden_size, bidirectional=bidirectional)
        if not bidirectional:
            self.lstm = nn.LSTM(2 * hidden_size, hidden_size, batch_first=True, num_layers=num_layers, bidirectional=False)
            self.out = nn.Linear(hidden_size, output_size)
        else:
            self.lstm = nn.LSTM(3 * hidden_size, hidden_size, batch_first=True, num_layers=num_layers, bidirectional=True)
            self.out = nn.Linear(hidden_size*2, output_size)
        self.bridge1 = nn.Linear(hidden_size, hidden_size)
        self.bridge2 = nn.Linear(hidden_size, hidden_size)

        self.bidirectional = bidirectional

        for param in self.lstm.named_parameters():
            layer_name, wt = param
            if 'bias' in layer_name:
                torch.nn.init.zeros_(wt)
            else:
                if '_hh_' in layer_name:
                    for idx in range(4):
                        wt = get_orthogonal_matrix(*wt.shape)
                else:
                    torch.nn.init.normal_(wt, mean=0.0, std=1.0) 
        
        for name, param in self.bridge1.named_parameters():
            if 'weight' in name:
                torch.nn.init.normal_(param)
            else:
                torch.nn.init.zeros_(param)

        for name, param in self.bridge2.named_parameters():
            if 'weight' in name:
                torch.nn.init.normal_(param)
            else:
                torch.nn.init.zeros_(param)

        for name, param in self.out.named_parameters():
            if 'weight' in name:
                torch.nn.init.normal_(param)
            else:
                torch.nn.init.zeros_(param)

    def forward(self, encoder_outputs, encoder_hidden, input_mask,
                target_tensor=None, SOS_token=0, max_len=10):
        # Teacher forcing if given a target_tensor, otherwise greedy.
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        #decoder_hidden = encoder_hidden # TODO: Consider bridge
        
        decoder_hidden = (self.bridge1(encoder_hidden[0]), self.bridge2(encoder_hidden[1]))
        decoder_outputs, attn_maps = [], []

        for i in range(max_len):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs, input_mask)
            decoder_outputs.append(decoder_output)
            attn_maps.append(attn_weights)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                topv, topi = decoder_output.data.topk(1)
                decoder_input = topi.squeeze(-1)

        decoder_outputs = torch.cat(decoder_outputs, dim=1) # [B, Seq, OutVocab]
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, attn_maps


    def forward_step(self, input, hidden, encoder_outputs, input_mask):
        # encoder_outputs: [B, Seq, D]
        query = [x.permute(1, 0, 2) for x in hidden] # [1, B, D] --> [B, 1, D]
        context, attn_weights = self.attention(query, encoder_outputs, input_mask)
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        attn = torch.cat((embedded, context), dim=2)
        
        output, hidden = self.lstm(attn, hidden)
        output = self.out(output)
        # output: [B, 1, OutVocab]
        return output, hidden, attn_weights


class EncoderDecoder(nn.Module):
    def __init__(self, hidden_size, input_vocab_size, output_vocab_size, num_layers=1, bidirectional=False):
        super(EncoderDecoder, self).__init__()
        self.encoder = EncoderRNN(input_vocab_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional)
        self.decoder = AttnDecoder(hidden_size, output_vocab_size, num_layers=num_layers, bidirectional=bidirectional)
        # self.decoder = DecoderRNN(hidden_size, output_vocab_size)

    def forward(self, inputs, input_mask, max_len):
        encoder_outputs, encoder_hidden = self.encoder(inputs)
        decoder_outputs, decoder_hidden, attn = self.decoder(
            encoder_outputs, encoder_hidden, input_mask, target_tensor=None, max_len=max_len)
        return decoder_outputs, decoder_hidden, attn

if __name__ == "__main__":
    
    NUM_LAYERS = 4
    BIDIRECTIONAL = False
    hidden_size = 256
    input_wordc = 56660
    output_wordc = 41101
    model = EncoderDecoder(hidden_size, input_wordc, output_wordc, num_layers=NUM_LAYERS, bidirectional=BIDIRECTIONAL).to(device)
    
    X, X_mask = torch.ones((16,20)).long().to(device), torch.ones((16,20)).long().to(device)
    
    out, hid, attn = model.forward(X, X_mask, 13)
    
    print (X.shape, '==>', out.shape)

