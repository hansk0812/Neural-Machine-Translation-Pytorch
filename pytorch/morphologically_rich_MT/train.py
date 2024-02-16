import os
import glob
import numpy as np

import torch
from torch import nn

from dataset import EnTamV2Dataset, BucketingBatchSampler
#from models.lstm import EncoderRNNLSTM, AttnDecoderRNNLSTM
from models.lstm_classifier import EncoderRNN, AttnDecoderRNN

from torch.utils.data import DataLoader

from timeit import default_timer as timer

torch.manual_seed(0)

def train_epoch(dataloader, model, optimizer, loss_fns, device):
    
    encoder, decoder = model
    ce_loss = loss_fns
    
    encoder_optimizer, decoder_optimizer, encoder_scheduler, decoder_scheduler = optimizer

    encoder.train()
    decoder.train()
    losses = 0

    for idx, (src, tgt) in enumerate(train_dataloader):
        src = src.to(device)
        tgt = tgt.long().to(device)

        encoder_outputs, encoder_hidden = encoder(src)
        # don't use teacher forcing
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_length=tgt.shape[1], target_tensor=None)
        
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        
        # decoder_outputs: B,L,C, tgt: B,L
        loss = ce_loss(decoder_outputs.reshape((-1, decoder_outputs.shape[-1])), tgt.reshape((-1)))
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        losses += loss.item()

    encoder_scheduler.step(loss)
    decoder_scheduler.step(loss)
    
    return losses / len(list(train_dataloader))

def evaluate(dataloader, model, loss_fns, device):

    encoder, decoder = model

    ce_loss = loss_fns
    
    encoder.eval()
    decoder.eval()
    losses = np.array([0])

    for src, tgt in val_dataloader:
        src = src.to(device)
        tgt = tgt.long().to(device)
        
        encoder_outputs, encoder_hidden = encoder(src)
        # don't use teacher forcing
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_length=tgt.shape[1], target_tensor=None)
        
        loss = ce_loss(decoder_outputs.reshape((-1, decoder_outputs.shape[-1])), tgt.reshape((-1)))
        #loss = ce_loss(decoder_outputs, tgt)
        
        losses[0] += loss.item()

    return losses[0] / len(list(val_dataloader))

if __name__ == "__main__":

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--nosymbols", action="store_true", help="Flag to remove symbols from dataset")
    ap.add_argument("--verbose", action="store_true", help="Flag to log things verbose")
    ap.add_argument("--morphemes", action="store_true", help="Flag to use morphological analysis on Tamil dataset")
    ap.add_argument("--batch_size", type=int, help="Num sentences per batch", default=256)
    ap.add_argument("--load_from_latest", action="store_true", help="Load from most recent epoch")
    ap.add_argument("--no_start_stop", action="store_true", help="Remove START and STOP tokens from sentences")
    ap.add_argument("--dropout_p", type=float, help="Change dropout probability from default=0.2", default=0.2)
    args = ap.parse_args()
 
    NUM_EPOCHS = 1500
   
    BATCH_SIZE = args.batch_size
    device = torch.device("cuda")

    train_dataset = EnTamV2Dataset("train", symbols=not args.nosymbols, verbose=args.verbose, 
                                   morphemes=args.morphemes, start_stop_tokens=not args.no_start_stop)
    
    eng_vocab, tam_vocab = train_dataset.return_vocabularies()
    
    val_dataset = EnTamV2Dataset("dev", symbols=not args.nosymbols, verbose=args.verbose, 
                                 morphemes=args.morphemes, vocabularies=(eng_vocab, tam_vocab), 
                                 start_stop_tokens=not args.no_start_stop)
    #test_dataset = EnTamV2Dataset("test", symbols=not args.nosymbols, verbose=args.verbose, morphemes=args.morphemes, vocabularies=(eng_vocab, tam_vocab))
    
    INPUT_SIZE = train_dataset.eng_embedding.shape[0]
    HIDDEN_DIM = train_dataset.eng_embedding.shape[1]
    OUTPUT_SIZE = train_dataset.tam_embedding.shape[0]
    
    train_bucketing_batch_sampler = BucketingBatchSampler(train_dataset.bucketing_indices, batch_size=BATCH_SIZE)
    val_bucketing_batch_sampler = BucketingBatchSampler(val_dataset.bucketing_indices, batch_size=BATCH_SIZE)
    
    train_dataloader = DataLoader(train_dataset, batch_sampler=train_bucketing_batch_sampler)
    val_dataloader = DataLoader(val_dataset, batch_sampler=val_bucketing_batch_sampler)

    #encoder = EncoderRNNLSTM(INPUT_SIZE, HIDDEN_DIM, weights=train_dataset.eng_embedding).to(device)
    #decoder = AttnDecoderRNNLSTM(HIDDEN_DIM, OUTPUT_SIZE, device, weights=train_dataset.tam_embedding).to(device)

    encoder = EncoderRNN(INPUT_SIZE, HIDDEN_DIM, 
                         weights=torch.tensor(train_dataset.eng_embedding), 
                         dropout_p=args.dropout_p).to(device)
    decoder = AttnDecoderRNN(HIDDEN_DIM, OUTPUT_SIZE, device=device, 
                             weights=torch.tensor(train_dataset.tam_embedding), 
                             dropout_p=args.dropout_p).to(device)
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=train_dataset.ignore_index) # Ignore PAD

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.00005, betas=(0.9, 0.98), eps=1e-9)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=0.00005, betas=(0.9, 0.98), eps=1e-9)
    
    encoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, mode='min', factor=0.2, patience=10, threshold=0.00001)
    decoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, mode='min', factor=0.2, patience=10, threshold=0.00001)
    
    if not os.path.isdir('trained_models'):
        best_val_loss = {"epoch": 1, "loss": np.inf}
        os.mkdir("trained_models")
    else:
        if args.load_from_latest:
            saved_model_path = sorted(glob.glob('trained_models/encoder_*.pt'), key=lambda x: int(x.split('epoch')[-1].split('_')[0]))[-1]
            load_epoch, load_loss = int(saved_model_path.split('epoch')[-1].split('_')[0]), float(saved_model_path.split('valloss')[-1].split('.')[0])
        else:
            saved_model_path = sorted(glob.glob('trained_models/encoder_*.pt'), key=lambda x: float(x.split('valloss')[-1].split('.')[0]))[0]
            load_epoch, load_loss = int(saved_model_path.split('epoch')[-1].split('_')[0]), float(saved_model_path.split('valloss')[-1].split('.')[0])
        
        best_val_loss = {"epoch": load_epoch+1, "loss": load_loss}
        
        print ("Load model from %s" % saved_model_path)
        encoder.load_state_dict(torch.load(saved_model_path))
        decoder.load_state_dict(torch.load(saved_model_path.replace("encoder", "decoder")))

    for epoch in range(best_val_loss["epoch"], NUM_EPOCHS+1):
        start_time = timer()
        train_loss = train_epoch(train_dataset, (encoder, decoder), (encoder_optimizer, decoder_optimizer, encoder_scheduler, decoder_scheduler), loss_fn, device)
        end_time = timer()
        
        with torch.no_grad():
            val_loss = evaluate(val_dataloader, (encoder, decoder), loss_fn, device)
        val_loss = np.mean(val_loss)
        
        if val_loss < best_val_loss["loss"]:

            best_val_loss["loss"] = val_loss
            best_val_loss["epoch"] = epoch

            torch.save(encoder.state_dict(), 'trained_models/encoder_epoch%d_valloss%f.pt' % (epoch, val_loss))
            torch.save(decoder.state_dict(), 'trained_models/decoder_epoch%d_valloss%f.pt' % (epoch, val_loss))
        
        if epoch % 10 == 0:
            torch.save(encoder.state_dict(), 'trained_models/encoder_epoch%d_valloss%f.pt' % (epoch, val_loss))
            torch.save(decoder.state_dict(), 'trained_models/decoder_epoch%d_valloss%f.pt' % (epoch, val_loss))

        print((f"Epoch: {epoch}, Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

