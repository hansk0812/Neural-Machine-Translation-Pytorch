import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False, dropout_p=0.3):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, num_layers=num_layers, bidirectional=bidirectional)

        self.dropout = nn.Dropout(dropout_p)

        for key in self.gru._parameters.keys():
            if "weight" in key:
                nn.init.sparse_(self.gru._parameters[key], sparsity=0.25)
            if "bias" in key:
                nn.init.uniform_(self.gru._parameters[key])

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
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
        
        for key in self.W1._parameters.keys():
            if 'weight' in key:
                nn.init.sparse_(self.W1._parameters[key], sparsity=0.25)
            if 'bias' in key:
                nn.init.uniform_(self.W1._parameters[key])
        for key in self.W2._parameters.keys():
            if 'weight' in key:
                nn.init.sparse_(self.W2._parameters[key], sparsity=0.25)
            if 'bias' in key:
                nn.init.uniform_(self.W2._parameters[key])
        for key in self.V._parameters.keys():
            if 'weight' in key:
                nn.init.sparse_(self.V._parameters[key], sparsity=0.25)
            if 'bias' in key:
                nn.init.uniform_(self.V._parameters[key])

    def forward(self, query, values, mask):
        # Additive attention
        
        query = query[:, -1:, :] if not self.bidirectional else torch.cat((query[:,-1:,:], query[:,-2:-1,:]), dim=-1) # last layer outputs

        scores = self.V(torch.tanh(self.W1(query) + self.W2(values)))
        scores = scores.squeeze(2).unsqueeze(1) # [B, M, 1] -> [B, 1, M]

        # Dot-Product Attention: score(s_t, h_i) = s_t^T h_i
        # Query [B, 1, D] * Values [B, D, M] -> Scores [B, 1, M]
        # scores = torch.bmm(query, values.permute(0,2,1))

        # Cosine Similarity: score(s_t, h_i) = cosine_similarity(s_t, h_i)
        # scores = F.cosine_similarity(query, values, dim=2).unsqueeze(1)

        # Mask out invalid positions.
        scores.data.masked_fill_(mask.unsqueeze(1) == 0, -float('inf'))

        # Attention weights
        alphas = F.softmax(scores, dim=-1)

        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, values)

        # context shape: [B, 1, D], alphas shape: [B, 1, M]
        return context, alphas


class AttnDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1, bidirectional=False, dropout_p=0.3):
        super(AttnDecoder, self).__init__()
        
        self.dropout = nn.Dropout(dropout_p)

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size, bidirectional=bidirectional)
        if not bidirectional:
            self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True, num_layers=num_layers, bidirectional=False)
            self.out = nn.Linear(hidden_size, output_size)
        else:
            self.gru = nn.GRU(3 * hidden_size, hidden_size, batch_first=True, num_layers=num_layers, bidirectional=True)
            self.out = nn.Linear(hidden_size*2, output_size)
        self.bridge = nn.Linear(hidden_size, hidden_size)

        self.bidirectional = bidirectional
        
        for key in self.gru._parameters.keys():
            if "weight" in key:
                nn.init.sparse_(self.gru._parameters[key], sparsity=0.25)
            if "bias" in key:
                nn.init.uniform_(self.gru._parameters[key])
            
        for key in self.out._parameters.keys():
            if 'weight' in key:
                nn.init.sparse_(self.out._parameters[key], sparsity=0.25)
            if 'bias' in key:
                nn.init.uniform_(self.out._parameters[key])

    def forward(self, encoder_outputs, encoder_hidden, input_mask,
                target_tensor=None, SOS_token=0, max_len=10):
        # Teacher forcing if given a target_tensor, otherwise greedy.
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        #decoder_hidden = encoder_hidden # TODO: Consider bridge
        
        decoder_hidden = self.bridge(encoder_hidden)
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
        query = hidden.permute(1, 0, 2) # [1, B, D] --> [B, 1, D]
        context, attn_weights = self.attention(query, encoder_outputs, input_mask)
        embedded = self.dropout(self.embedding(input))
        attn = torch.cat((embedded, context), dim=2)
        
        output, hidden = self.gru(attn, hidden)
        output = self.out(output)
        # output: [B, 1, OutVocab]
        return output, hidden, attn_weights


class EncoderDecoder(nn.Module):
    def __init__(self, hidden_size, input_vocab_size, output_vocab_size, num_layers=1, bidirectional=False, dropout_p=0.3):
        super(EncoderDecoder, self).__init__()
        self.encoder = EncoderRNN(input_vocab_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional, dropout_p=dropout_p)
        self.decoder = AttnDecoder(hidden_size, output_vocab_size, num_layers=num_layers, bidirectional=bidirectional, dropout_p=dropout_p)
        # self.decoder = DecoderRNN(hidden_size, output_vocab_size)

    def forward(self, inputs, input_mask, max_len):
        encoder_outputs, encoder_hidden = self.encoder(inputs)
        decoder_outputs, decoder_hidden, attn = self.decoder(
            encoder_outputs, encoder_hidden, input_mask, target_tensor=None, max_len=max_len)
        return decoder_outputs, decoder_hidden, attn

if __name__ == "__main__":

    import numpy as np

    NUM_LAYERS = 3
    BIDIRECTIONAL = False
    hidden_size = 256
    input_wordc = 56660
    output_wordc = 41101
    model = EncoderDecoder(hidden_size, input_wordc, output_wordc, num_layers=NUM_LAYERS, bidirectional=BIDIRECTIONAL).to(device)
    
    X, X_mask = torch.ones((16,20)).long().to(device), torch.ones((16,20)).long().to(device)
    
    out, hid, attn = model.forward(X, X_mask, 13)
    
    print (X.shape, '==>', out.shape)
    
    attn = np.array([x.detach().cpu().numpy() for x in attn]) # target_len, batch_size, 1, input_len
    
    attn = attn[:,:,0,:]
    for idx in range(16):
        attn_final = attn[:,idx,:]
        print (attn_final.shape)
