import torch
from torch import nn
from torch.nn import functional as F

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout_p=0.1, weights=None):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size, _weight=weights)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input)).float()
        output, hidden = self.gru(embedded)
        return output, hidden

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        bidirectional_index = query.shape[1]//2 - 1
        query = query[:,-1:,:] + query[:,bidirectional_index:bidirectional_index+1,:] # take last layer activations only
        #query = torch.sum(query, axis=1).unsqueeze(1) # add activations across num_layers
        
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, device, num_layers=2, dropout_p=0.1, weights=None):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size, _weight=weights)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)
        self.device = device

    def forward(self, encoder_outputs, encoder_hidden, target_length, target_tensor=None):
        SOS_token = 0
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=self.device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(target_length):
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
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input)).float()

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights

if __name__ == "__main__":

    from dataset import EnTamV2Dataset
    train_dataset = EnTamV2Dataset("train", symbols=True, verbose=False, morphemes=False)

    INPUT_SIZE=train_dataset.eng_embedding.shape[0]
    HIDDEN_DIM=train_dataset.eng_embedding.shape[1]
    OUTPUT_SIZE=train_dataset.tam_embedding.shape[0]
    device = torch.device("cpu")

    encoder = EncoderRNN(INPUT_SIZE, HIDDEN_DIM, weights=torch.tensor(train_dataset.eng_embedding).to(device))
    decoder = AttnDecoderRNN(HIDDEN_DIM, OUTPUT_SIZE, weights=torch.tensor(train_dataset.tam_embedding).to(device))

    x = torch.ones((64,30)).long()
    y = torch.ones((64,20)).long()

    encoder_outputs, encoder_hidden = encoder(x)
    decoder_outputs, decoder_hidden, attn = decoder(encoder_outputs, encoder_hidden, target_length=y.shape[1])

    print (decoder_outputs.shape)
