import torch
from torch import nn
from torch.nn import functional as F

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        
        print ("Encoder: %d layers" % num_layers)
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, num_layers=num_layers, bidirectional=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size*2, hidden_size*2)
        self.Ua = nn.Linear(hidden_size*2, hidden_size*2)
        self.Va = nn.Linear(hidden_size*2, 1)

    def forward(self, query, keys):
        # query: (batch_size, num_layers, hidden_size)
        query = query.view((query.shape[0], 2, -1, query.shape[-1]))
        query = torch.cat((query[:,0:1,-1,:], query[:,1:2,-1,:]), dim=-1)
        
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, device, num_layers, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()

        print ("Decoder: %d layers" % num_layers)
        self.device = device

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(3*hidden_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2*hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_len, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=self.device).fill_(0)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(target_len):
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
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights

class Seq2Seq(nn.Module):
    
    def __init__(self, encoder, decoder):

        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_tensor, target_tensor, target_len, device):
        
        encoder_outputs, encoder_hidden = self.encoder(input_tensor)
        decoder_outputs, decoder_hidden, attentions = self.decoder(encoder_outputs, encoder_hidden, target_len, target_tensor=target_tensor)
        
        return decoder_outputs, attentions

def init_weights(m):
    for name, param in m.named_parameters():
        if "weight" in name:
            nn.init.kaiming_uniform_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0.5)

if __name__ == "__main__":
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    input_dim = 66000
    hidden_size = 256
    output_dim = 44000

    batch_size = 16

    seq1_len = 30
    seq2_len = 20

    num_layers = 6

    encoder = EncoderRNN(
        input_dim,
        hidden_size,
        num_layers,
        0.2
    )

    decoder = AttnDecoderRNN(
        hidden_size,
        output_dim,
        device,
        num_layers,
        0.2
    )

    x = torch.ones((64, 30)).long().to(device)
    y = torch.ones((64, 20)).long().to(device)

    net = Seq2Seq(encoder, decoder).to(device)

    output, hidden = net(x, y, 20, device)
    
    print (x.shape, '-->', output.shape, hidden.shape) 

