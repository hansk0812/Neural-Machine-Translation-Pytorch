import os
import glob
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np

from models.gru import EncoderDecoder

from data.utils import get_sentences_from_file, BucketingBatchSamplerReplace as BucketingBatchSampler
from bilingual_sets.entam import EnTam, BucketingBatchSampler

from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib
matplotlib.use("agg")

prop = FontProperties()
prop.set_file('./utils/Tamil001.ttf')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 56

train_dataset = EnTam("dataset/corpus.bcn.train.en", "dataset/corpus.bcn.train.ta", start_stop=False, bucketing_language_sort="l2", cache_id=0, morphemes=True)
vocabs = train_dataset.return_vocabularies()
word2vecs = train_dataset.return_word2vecs()
val_dataset = EnTam("dataset/corpus.bcn.dev.en", "dataset/corpus.bcn.dev.ta", start_stop=False, bucketing_language_sort="l2", vocabularies=vocabs, word2vecs=word2vecs, cache_id=1, morphemes=True)
test_dataset = EnTam("dataset/corpus.bcn.test.en", "dataset/corpus.bcn.test.ta", start_stop=False, bucketing_language_sort="l2", vocabularies=vocabs, word2vecs=word2vecs, cache_id=2, morphemes=True)

bucketing_batch_sampler = BucketingBatchSampler(train_dataset.bucketer.bucketing_indices, batch_size=batch_size, verbose=True)
train_dataloader = DataLoader(train_dataset, batch_sampler=bucketing_batch_sampler)

bucketing_batch_sampler = BucketingBatchSampler(val_dataset.bucketer.bucketing_indices, batch_size=batch_size, verbose=True)
val_dataloader = DataLoader(val_dataset, batch_sampler=bucketing_batch_sampler)

bucketing_batch_sampler = BucketingBatchSampler(test_dataset.bucketer.bucketing_indices, batch_size=1, verbose=True)
test_dataloader = DataLoader(test_dataset, batch_sampler=bucketing_batch_sampler)

PAD_idx = train_dataset.pad_idx_2

hidden_size = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize_attn_map(map_tensor, x, y_pred, index, attention_maps_str):

    # map_tensor: [M, N], M: num decoder outputs, N: num encoder outputs
    # x: Input sentence as string
    # y_pred: Predicted sentence as string

    if not os.path.isdir("attn_maps/%s" % attention_maps_str):
        os.makedirs('attn_maps/%s' % attention_maps_str)

    Y = [i for i in y_pred.split(' ') if i != '']
    X = [i for i in x.split(' ') if i != '']
    
    attn_map = map_tensor # decoder len x encoder len
    
    try:
        fig
        ax
    except Exception:
        fig, ax = plt.subplots()
    
    im = ax.imshow(attn_map[:,0,:])
    
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
    plt.savefig('attn_maps/%s/%d.png' % (attention_maps_str, index))
    plt.close()

def train(train_dataloader, val_dataloader, model, epoch, n_epochs, learning_rate=0.00003, attention_maps_str=""):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 32)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.75, patience=40)

    #criterion = nn.CrossEntropyLoss() # - attn maps more intuitive with PADs, faster convergence
    criterion = nn.NLLLoss()

    img_id = 0
    
    for epoch in range(epoch, n_epochs + 1):
        loss = 0
        for iter, batch in enumerate(train_dataloader):
            # Batch tensors: [B, SeqLen]
            input_tensor  = batch[0].to(device)
            input_mask    = batch[1].to(device)
            target_tensor = batch[2].to(device)
            loss += train_step(input_tensor, input_mask, target_tensor,
                               model, optimizer, criterion)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # clip gradients
        #scheduler.step()
        print('Epoch {} Loss {}'.format(epoch, loss / iter))

        with torch.no_grad():
            val_loss, min_loss = 0, np.inf
            for iter, batch in enumerate(val_dataloader):
                input_tensor  = batch[0].to(device)
                input_mask    = batch[1].to(device)
                target_tensor = batch[2].to(device)
    
                decoder_outputs, decoder_hidden, attn_wts = model(input_tensor, input_mask, max_len=target_tensor.shape[1])
                # Collapse [B, Seq] dimensions for NLL Loss
                val_loss += criterion(
                    decoder_outputs.view(-1, decoder_outputs.size(-1)), # [B, Seq, OutVoc] -> [B*Seq, OutVoc]
                    target_tensor.view(-1) # [B, Seq] -> [B*Seq]
                )
            val_loss = val_loss / iter

        # serialization
        if epoch % 10 == 0 or val_loss < min_loss: 
            
            attn_wts = np.array([x.cpu().numpy() for x in attn_wts])
            for idx in range(attn_wts.shape[1]):
                input_sent = train_dataset.indices_to_words(input_tensor[idx].cpu().numpy(), language='en')
            
                topv, topi = decoder_outputs.topk(1)
                decoded_ids = topi.squeeze()
                output_sent = train_dataset.indices_to_words(decoded_ids[idx].cpu().numpy(), language='ta')
            
                target_sent = train_dataset.indices_to_words(target_tensor[idx].cpu().numpy(), language='ta')
                
                visualize_attn_map(attn_wts[:,idx,...], input_sent, output_sent, iter * img_id, attention_maps_str) 
                img_id += 1
            
            torch.save(model.state_dict(), "trained_models/%f_epoch_%d.pt" % (val_loss, epoch))
            min_loss = val_loss

def train_step(input_tensor, input_mask, target_tensor, model,
               optimizer, criterion):
    optimizer.zero_grad()
    decoder_outputs, decoder_hidden, attn_wts = model(input_tensor, input_mask, max_len=target_tensor.shape[1])

    # Collapse [B, Seq] dimensions for NLL Loss
    loss = criterion(
        decoder_outputs.view(-1, decoder_outputs.size(-1)), # [B, Seq, OutVoc] -> [B*Seq, OutVoc]
        target_tensor.view(-1) # [B, Seq] -> [B*Seq]
    )

    loss.backward()
    optimizer.step()
    return loss.item()

def ids2words(lang, ids):
    return [lang.index2word[idx] for idx in ids]

def greedy_decode(model, training_data_obj, dataloader, device, attention_maps_str):
    total_idx = 0
    bleu_scores = []
    with torch.no_grad():
        
        dataloader = iter(dataloader)
        while True:
            try:
                batch = next(dataloader)
            except StopIteration:
                break
            input_tensor  = batch[0].to(device)
            input_mask    = batch[1].to(device)
            target_tensor = batch[2].to(device)

            decoder_outputs, decoder_hidden, attn_wts = model(input_tensor, input_mask, max_len=target_tensor.shape[1])
            topv, topi = decoder_outputs.topk(1)
            decoded_ids = topi.squeeze()
            attn_wts = [x.cpu().numpy() for x in attn_wts] # list length = target length [B,1,X_len]
            
            for idx in range(input_tensor.size(0)):
                input_sent = train_dataset.indices_to_words(input_tensor[idx].cpu().numpy(), language='en')
                if len(decoded_ids.shape) == 1:
                    output_sent = train_dataset.indices_to_words(decoded_ids.cpu().numpy(), language='ta')
                else:    
                    output_sent = train_dataset.indices_to_words(decoded_ids[idx].cpu().numpy(), language='ta')
                target_sent = train_dataset.indices_to_words(target_tensor[idx].cpu().numpy(), language='ta')
                #print('Input:  {}'.format(input_sent), end='\r')
                #print ()
                #print('Target: {}'.format(target_sent), end='\r')
                #print ()
                #print('Output: {}'.format(output_sent), end='\r')
                #print ()

                visualize_attn_map(attn_wts[idx], input_sent, output_sent, total_idx * input_tensor.size(0) + idx, attention_maps_str) 
                total_idx += 1
                
                bleu_score = sentence_bleu(
                                 [target_sent.split(' ')],
                                  output_sent.split(' '))
                bleu_scores.append(bleu_score)
                print (bleu_score)

    print ('BLEU score:', np.mean(bleu_scores))

if __name__ == '__main__':
    
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("attention_maps_str", help='String folder to save attn weights image')
    ap.add_argument("--load_from_latest", help='Load most recent checkpoint (bool)', action="store_true")
    args = ap.parse_args()

    hidden_size = 256
    input_wordc = len(train_dataset.l1_vocab.sorted_tokens)
    output_wordc = len(train_dataset.l2_vocab.sorted_tokens)
    
    model = EncoderDecoder(hidden_size, input_wordc, output_wordc, num_layers=1).to(device)
    
    model_chkpts = glob.glob("trained_models/*")
    if len(model_chkpts) > 0:

        if not args.load_from_latest:
            model_chkpts = sorted(model_chkpts, reverse=True, key=lambda x: float(x.split('/')[-1].split('_epoch_')[0]))
        else:
            model_chkpts = sorted(model_chkpts, key=lambda x: float(x.split('/')[-1].split('_epoch_')[-1].split('.pt')[0]))

        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_chkpts[-1]))
        else:
            model.load_state_dict(torch.load(model_chkpts[-1], map_location=device))

        epoch = int(model_chkpts[-1].split('/')[-1].split('_epoch_')[1].split('.pt')[0])
        print ("Loaded model from file: %s" % model_chkpts[-1])
    else:
        try:
            os.makedirs("trained_models")
        except Exception:
            pass
        epoch = 1

    if not os.path.isdir("attn_maps/" + args.attention_maps_str):
        os.makedirs("attn_maps/" + args.attention_maps_str)
    
    train(train_dataloader, val_dataloader, model, epoch, n_epochs=1500, attention_maps_str=args.attention_maps_str)
    greedy_decode(model, train_dataset, test_dataloader, device=device, attention_maps_str=args.attention_maps_str)
