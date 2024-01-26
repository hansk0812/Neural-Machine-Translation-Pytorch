import torch
from torch import nn
from torch.nn import functional as F

class EncoderRNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super().__init__()
        self.hidden_size = hidden_size

        self.ff = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=3, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.ff(input))
        output, hidden = self.lstm(embedded)
        return output, hidden

class BahdanauAttentionLSTM(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        query = query[0] + query[1] # h_n + c_n (h_n alone is better)
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoderRNNLSTM(nn.Module):
    def __init__(self, hidden_size, output_size, device, dropout_p=0.1):
        super().__init__()

        self.device = device
        self.output_size = output_size

        self.ff = nn.Linear(output_size, hidden_size)
        self.attention = BahdanauAttentionLSTM(hidden_size)
        self.lstm = nn.LSTM(2 * hidden_size, hidden_size, num_layers=3, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, max_length, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        
        SOS_token = 0 # start with zeros because of lack of previous state
        decoder_input = torch.empty(batch_size, 1, self.output_size, dtype=torch.float, device=self.device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(max_length):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                #_, topi = decoder_output.topk(1)
                #decoder_input = topi.squeeze(-1).detach()  # detach from history as input
                # not classifying for word2vec
                decoder_input = decoder_input + decoder_output # previous output fed as LSTM input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.ff(input))

        query = [torch.sum(x.permute(1, 0, 2), axis=1).unsqueeze(1) for x in hidden]
        context, attn_weights = self.attention(query, encoder_outputs)

        input_gru = torch.cat((embedded, context), dim=2)
        
        output, hidden = self.lstm(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights

if __name__ == "__main__":

    word2vec_vector_size = 100
    hidden_size = 128
    bucket = [30,20]
    device = torch.device("cuda")

    encoder = EncoderRNNLSTM(word2vec_vector_size, hidden_size).to(device)
    decoder = AttnDecoderRNNLSTM(hidden_size, word2vec_vector_size, device=device).to(device)
    
    input_tensor = torch.ones((64,bucket[0],word2vec_vector_size)).to(device)
    target_tensor = torch.ones((64,bucket[1],word2vec_vector_size)).to(device)
    encoder_outputs, encoder_hidden = encoder(input_tensor)
    print ('encoder outputs', encoder_outputs.shape,'encoder hidden',  [x.shape for x in encoder_hidden])
    #print ([x.shape for x in encoder_outputs], [x.shape for x in encoder_hidden])

    decoder_outputs, decoder_hidden, attn = decoder(encoder_outputs, encoder_hidden, max_length=bucket[1], target_tensor=None)#target_tensor)
    print (decoder_outputs.shape, [x.shape for x in decoder_hidden], attn.shape)
