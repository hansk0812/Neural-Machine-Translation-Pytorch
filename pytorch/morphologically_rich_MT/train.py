import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from bilingual_sets.entam import EnTam as EnTamV2Dataset
from data.utils import BucketingBatchSamplerReplace as BucketingBatchSampler

from models.gru_seq2seq import Seq2Seq, Encoder, Decoder, Attention, init_weights

from nltk.translate.bleu_score import sentence_bleu

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
matplotlib.use("agg")

prop = FontProperties()
prop.set_file('./utils/Tamil001.ttf')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize_attn_map(map_tensor, x, y_pred, index):

    # map_tensor: [M, N], M: num decoder outputs, N: num encoder outputs
    # x: Input sentence as string
    # y_pred: Predicted sentence as string
    
    Y = [i for i in y_pred.split(' ') if i != '']
    X = [i for i in x.split(' ') if i != '']
    
    attn_map = map_tensor.cpu().numpy() # decoder len x encoder len
    
    try:
        fig
        ax
    except Exception:
        fig, ax = plt.subplots()
    
    im = ax.imshow(attn_map)
    
    # Show all ticks and label them with the respective list entries
    # Compatibility issues because of matplotlib version
    ax.set_xticks(np.arange(len(X)))#, labels=X)
    ax.set_yticks(np.arange(len(Y)))#, labels=Y, fontproperties=prop)

    ax.set_xticklabels(X)
    ax.set_yticklabels(Y, fontproperties=prop)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    #for i in range(len(Y)):
    #    for j in range(len(X)):
    #        text = ax.text(j, i, attn_map[i, j],
    #                       ha="center", va="center", color="w")

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Attention weights", rotation=-90, va="bottom")

    ax.set_title("Bahdanau Attention")
    fig.tight_layout()
    
    #plt.show()
    plt.savefig('%d.png' % index)

def train(train_dataloader, val_dataloader, model, n_epochs, PAD_idx, start_epoch, learning_rate=0.0003):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #criterion = nn.NLLLoss(ignore_index=PAD_idx)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_idx)

    model.train()
    
    if not os.path.isdir("trained_models"):
        os.mkdir("trained_models")

    # sanity check: all data
    #for batch in train_dataloader:
    #    input_tensor, target_tensor, input_mask = batch
 
    #    decoder_outputs, decoder_hidden, attn_map = model(input_tensor, input_mask, max_len=target_tensor.shape[1])

    #    print (decoder_outputs.shape)
    #    topv, topi = decoder_outputs.topk(1)
    #    print (topv.shape, top1.shape)
    #    decoded_ids = topi.squeeze()

    #   for idx in range(input_tensor.size(0)):
    #       input_sent = train_dataset.vocab_indices_to_sentence(input_tensor[idx], "en")
    #        output_sent = train_dataset.vocab_indices_to_sentence(decoded_ids[idx], "ta")
    #        target_sent = train_dataset.vocab_indices_to_sentence(target_tensor[idx], "ta")

    #        visualize_attn_map(attn_map[idx], input_sent, output_sent, idx) 
            
    #        print (output_sent)
    #        print (target_sent)
    #        print ()
    
    min_loss = float('inf')
    for epoch in range(start_epoch, n_epochs + 1):
        loss = 0
        for iter, batch in enumerate(train_dataloader):
            # Batch tensors: [B, SeqLen]
            input_tensor  = batch[0].transpose(1,0).to(device)
            target_tensor = batch[1].transpose(1,0).type(torch.LongTensor).to(device)

            loss += train_step(input_tensor, target_tensor,
                               model, optimizer, criterion)
        print('Epoch {} Loss {}'.format(epoch, loss / iter))
        
        model.eval()
        val_loss = validate(model, val_dataloader, criterion)
        model.train()

        epoch_loss = loss/float(len(train_dataloader)*len(batch[0]))
        # serialization
        if epoch % 10 == 0 or val_loss < min_loss: #epoch_loss < min_loss:
            torch.save(model.state_dict(), "trained_models/IBM_epoch%d_loss%.5f.pt" % (epoch, val_loss))
            min_loss = val_loss

        # add gradient clipping
        for param in model._parameters:
            model._parameters[param] = torch.clip(model._parameters[param], min=-5, max=5) 

def train_step(input_tensor, target_tensor, model,
               optimizer, criterion):
    optimizer.zero_grad()

    decoder_outputs, attn_map = model(input_tensor, target_tensor, 0.2 ) #target_tensor)
    
    print (target_tensor.shape)
    # Collapse [B, Seq] dimensions for NLL Loss
    loss = criterion(
        decoder_outputs.view(-1, decoder_outputs.size(-1)), # [B, Seq, OutVoc] -> [B*Seq, OutVoc]
        target_tensor.contiguous().view(-1) # [B, Seq] -> [B*Seq]
    )

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

    optimizer.step()
    return loss.item()

def test(model, dataloader):
    bleu_scores = []
    with torch.no_grad():
        for batch in dataloader:
            input_tensor  = batch[0].transpose(1,0).to(device)
            target_tensor = batch[1].transpose(1,0).to(device)

            decoder_outputs, attn_map = model(input_tensor, input_mask, max_len=target_tensor.shape[1])
            decoder_outputs = decoder_outputs.transpose(1, 0)

            topv, topi = decoder_outputs.topk(1)
            decoded_ids = topi.squeeze()

            input_tensor = input_tensor.transpose(1, 0)
            target_tensor = input_tensor.transpose(1, 0)

            for idx in range(input_tensor.size(0)):
                input_sent = train_dataset.vocab_indices_to_sentence(input_tensor[idx], "en")
                output_sent = train_dataset.vocab_indices_to_sentence(decoded_ids[idx], "ta")
                target_sent = train_dataset.vocab_indices_to_sentence(target_tensor[idx], "ta")

                visualize_attn_map(attn_map[idx], input_sent, output_sent, idx) 
                
                print (output_sent)

                output_sent = output_sent.split(' ')
                last, ctx = output_sent[-1], 1
                while(output_sent[-2] == last):
                    output_sent.pop()
                    ctx += 1
                
                if ctx > 1:
                    output_sent = ' '.join(output_sent + ["END"] + ["PAD"]*(ctx-1))
                
                print (output_sent)
                
                bleu_scores.append(sentence_bleu([target_sent], output_sent))

        print ("Test BLEU score average = %.5f" % np.mean(bleu_scores))

def validate(model, dataloader, criterion):
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            input_tensor  = batch[0].transpose(1,0).to(device)
            target_tensor = batch[1].transpose(1,0).type(torch.LongTensor).to(device)

            decoder_outputs, attn_maps = model(input_tensor, target_tensor, 0.)
            loss = criterion(
                    decoder_outputs.view(-1, decoder_outputs.size(-1)), # [B, Seq, OutVoc] -> [B*Seq, OutVoc]
                    target_tensor.view(-1) # [B, Seq] -> [B*Seq]
                )
            losses.append(loss)
        
        loss_val = torch.mean(torch.tensor(losses))
        print ("\n\t\tVal set loss: %.7f\n" % loss_val)

    return loss_val

def init_weights(m):
    for name, param in m.named_parameters():
        if "weight" in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

if __name__ == '__main__':
    
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", "-v", help="Verbose flag for dataset stats", action="store_true")
    ap.add_argument("--nosymbols", "-ns", help="Symbols flag for eliminating symbols from dataset", action="store_true")
    ap.add_argument("--no_start_stop", "-nss", help="Remove START and STOP tokens", action="store_true")
    ap.add_argument("--no_linear", "-nl", help="Remove FF layers", action="store_true")
    ap.add_argument("--morphemes", "-m", help="Morphemes flag for morphological analysis", action="store_true")
    ap.add_argument("--batch_size", "-b", help="Batch size (int)", type=int, default=64)
    ap.add_argument("--num_layers", "-n", help="Number of RNN layers (int)", type=int, default=3)
    ap.add_argument("--n_epochs", "-ne", help="Number of training epochs (int)", type=int, default=500)
    ap.add_argument("--dropout_p", "-d", help="Dropout probability (float)", type=float, default=0.2)
    ap.add_argument("--test", "-t", help="Flag for testing over test set", action="store_true")
    ap.add_argument("--load_from_latest", "-ll", help="Flag for loading latest checkpoint", action="store_true")
    args = ap.parse_args()
    
    train_dataset = EnTamV2Dataset("dataset/corpus.bcn.train.en", "dataset/corpus.bcn.train.ta", bucketing_language_sort="l2", cache_id=0)

    eng_vocab, tam_vocab = train_dataset.return_vocabularies()
    word2vecs = train_dataset.return_word2vecs()
    PAD_idx = train_dataset.pad_idx
    
    if not args.test:
        val_dataset = EnTamV2Dataset("dataset/corpus.bcn.dev.en", "dataset/corpus.bcn.dev.ta", bucketing_language_sort="l2", 
                            vocabularies=[eng_vocab, tam_vocab], word2vecs=word2vecs, cache_id=1)
    else:
        test_dataset = EnTamV2Dataset("dataset/corpus.bcn.test.en", "dataset/corpus.bcn.test.ta", bucketing_language_sort="l2", 
                            vocabularies=[eng_vocab, tam_vocab], word2vecs=word2vecs, cache_id=2)
    
    from torch.utils.data import DataLoader
    
    if not args.test:
        train_bucketing_batch_sampler = BucketingBatchSampler(train_dataset.bucketer.bucketing_indices, batch_size=args.batch_size)
        train_dataloader = DataLoader(train_dataset, batch_sampler=train_bucketing_batch_sampler)
        
        val_bucketing_batch_sampler = BucketingBatchSampler(val_dataset.bucketer.bucketing_indices, batch_size=args.batch_size)
        val_dataloader = DataLoader(val_dataset, batch_sampler=val_bucketing_batch_sampler)
    else:
        test_bucketing_batch_sampler = BucketingBatchSampler(test_dataset.bucketer.bucketing_indices, batch_size=args.batch_size)
        test_dataloader = DataLoader(test_dataset, batch_sampler=test_bucketing_batch_sampler)
    
    hidden_size = 256
    input_dim = len(eng_vocab)
    output_dim = len(tam_vocab)

    encoder_embedding_dim = hidden_size
    decoder_embedding_dim = hidden_size
    encoder_hidden_dim = 512
    decoder_hidden_dim = 512
    encoder_dropout = args.dropout_p
    decoder_dropout = args.dropout_p

    attention = Attention(encoder_hidden_dim, decoder_hidden_dim)

    encoder = Encoder(
        input_dim,
        encoder_embedding_dim,
        encoder_hidden_dim,
        decoder_hidden_dim,
        encoder_dropout,
        args.num_layers
    )

    decoder = Decoder(
        output_dim,
        decoder_embedding_dim,
        encoder_hidden_dim,
        decoder_hidden_dim,
        decoder_dropout,
        attention,
        args.num_layers
    )

    model = Seq2Seq(encoder, decoder, device).to(device)

    model.apply(init_weights)

    train_dataset.l1_embedding = torch.tensor(train_dataset.l1_embedding).to(device)
    train_dataset.l2_embedding = torch.tensor(train_dataset.l2_embedding).to(device)

    import glob
    import os

    model_chkpts = glob.glob("trained_models/*")
    if len(model_chkpts) > 0:

        if not args.load_from_latest:
            model_chkpts = sorted(model_chkpts, reverse=True, key=lambda x: float(x.split('loss')[1].split('.pt')[0]))
        else:
            model_chkpts = sorted(model_chkpts, key=lambda x: float(x.split('epoch')[1].split('_')[0]))

        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_chkpts[-1]))
        else:
            model.load_state_dict(torch.load(model_chkpts[-1], map_location=device))

        epoch = int(model_chkpts[-1].split('epoch')[1].split('_')[0])
        print ("Loaded model from file: %s" % model_chkpts[-1])
    else:
        epoch = 1
    
    if not args.test:
        train(train_dataloader, val_dataloader, model, n_epochs=args.n_epochs, PAD_idx=PAD_idx, start_epoch=epoch)
    else:
        test(model, test_dataloader)
