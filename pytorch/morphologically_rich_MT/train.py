import os
import glob
import numpy as np

import torch
from torch import nn

from dataset import EnTamV2Dataset, BucketingBatchSampler

#from models.lstm import EncoderRNNLSTM, AttnDecoderRNNLSTM

from models.lstm_classifier import EncoderRNN, AttnDecoderRNN
#from models.gru_classifier import EncoderRNN, AttnDecoderRNN

#from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq #TODO

from torch.utils.data import DataLoader

from timeit import default_timer as timer

torch.manual_seed(0)

def train_epoch(dataloader, model, optimizer, loss_fns, device):
    
    ce_loss = loss_fns
    
    if len(optimizer) == 4:
        encoder, decoder = model
        encoder_optimizer, decoder_optimizer, encoder_scheduler, decoder_scheduler = optimizer
        encoder.train()
        decoder.train()
    else:
        optimizer, scheduler = optimizer
        model.train()

    losses = 0

    for idx, (src, tgt) in enumerate(train_dataloader):
        src = src.to(device)
        tgt = tgt.long().to(device)
        
        if len(optimizer) == 4:
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
        
            encoder_outputs, encoder_hidden = encoder(src)
            # don't use teacher forcing
            decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_length=tgt.shape[1], target_tensor=None)
        
        else:
            optimizer.zero_grad()
            #decoder_outputs = model(
            #TODO

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
    
    ce_loss = loss_fns
    losses = np.array([0])
    
    if isinstance(model, tuple):
        encoder, decoder = model
    
        encoder.eval()
        decoder.eval()

    for src, tgt in val_dataloader:
        src = src.to(device)
        tgt = tgt.long().to(device)
        
        if isinstance(model, tuple):
            encoder_outputs, encoder_hidden = encoder(src)
            # don't use teacher forcing
            decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_length=tgt.shape[1], target_tensor=None)
        #else:
        #    pass
        #    #TODO

        loss = ce_loss(decoder_outputs.reshape((-1, decoder_outputs.shape[-1])), tgt.reshape((-1)))
        #loss = ce_loss(decoder_outputs, tgt)
        
        losses[0] += loss.item()

    return losses[0] / len(list(val_dataloader))

if __name__ == "__main__":

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("epochs", type=int, help="Number of epochs of training")
    ap.add_argument("--nosymbols", action="store_true", help="Flag to remove symbols from dataset")
    ap.add_argument("--verbose", action="store_true", help="Flag to log things verbose")
    ap.add_argument("--morphemes", action="store_true", help="Flag to use morphological analysis on Tamil dataset")
    ap.add_argument("--batch_size", type=int, help="Num sentences per batch", default=256)
    ap.add_argument("--load_from_latest", action="store_true", help="Load from most recent epoch")
    ap.add_argument("--no_start_stop", action="store_true", help="Remove START and STOP tokens from sentences")
    ap.add_argument("--dropout_p", type=float, help="Change dropout probability from default=0.2", default=0.2)
    ap.add_argument("--pytorch_seq2seq", action="store_true", help="Use pytorch-seq2seq from IBM for model") #TODO
    args = ap.parse_args()

    assert not args.pytorch_seq2seq or args.pytorch_seq2seq and not args.no_start_stop
 
    NUM_EPOCHS = args.epochs
   
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

    if not args.pytorch_seq2seq:
        #encoder = EncoderRNNLSTM(INPUT_SIZE, HIDDEN_DIM, weights=train_dataset.eng_embedding).to(device)
        #decoder = AttnDecoderRNNLSTM(HIDDEN_DIM, OUTPUT_SIZE, device, weights=train_dataset.tam_embedding).to(device)
        encoder = EncoderRNN(INPUT_SIZE, HIDDEN_DIM, num_layers=4,
                             weights=torch.tensor(train_dataset.eng_embedding), 
                             dropout_p=args.dropout_p).to(device)
        decoder = AttnDecoderRNN(HIDDEN_DIM, OUTPUT_SIZE, device=device, num_layers=4,
                             weights=torch.tensor(train_dataset.tam_embedding), 
                             dropout_p=args.dropout_p).to(device)
    
    else:
        encoder = EncoderRNN(INPUT_SIZE, train_dataset.buckets[-1][0], HIDDEN_DIM,
                                 bidirectional=True, variable_lengths=True)
        decoder = DecoderRNN(OUTPUT_SIZE, train_dataset.buckets[-1][1], HIDDEN_DIM * 2,
                                 dropout_p=0.2, use_attention=True, bidirectional=True,
                                 eos_id=train_dataset.eos_idx, sos_id=train_dataset.bos_idx)
        seq2seq = Seq2seq(encoder, decoder)
    
    loss_fn = nn.CrossEntropyLoss() # don't ignore PAD
    #loss_fn = nn.CrossEntropyLoss(ignore_index=train_dataset.ignore_index) # Ignore PAD
    
    if not args.pytorch_seq2seq:
        encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.00005, betas=(0.9, 0.98), eps=1e-9)
        decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=0.00005, betas=(0.9, 0.98), eps=1e-9)
    
        encoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, mode='min', factor=0.4, patience=5, threshold=0.000001)
        decoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, mode='min', factor=0.4, patience=5, threshold=0.000001)
    
    else:
        optimizer = torch.optim.Adam(seq2seq.parameters(), lr=0.00005, betas=(0.9, 0.98), eps=1e-9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, threshold=0.00001)
    
    if not args.pytorch_seq2seq:
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
    else:
        if not os.path.isdir('trained_models'):
            best_val_loss = {"epoch": 1, "loss": np.inf}
            os.mkdir("trained_models")
        else:
            if args.load_from_latest:
                saved_model_path = sorted(glob.glob('trained_models/seq2seq_*.pt'), key=lambda x: int(x.split('epoch')[-1].split('_')[0]))[-1]
                load_epoch, load_loss = int(saved_model_path.split('epoch')[-1].split('_')[0]), float(saved_model_path.split('valloss')[-1].split('.')[0])
            else:
                saved_model_path = sorted(glob.glob('trained_models/seq2seq_*.pt'), key=lambda x: float(x.split('valloss')[-1].split('.')[0]))[0]
                load_epoch, load_loss = int(saved_model_path.split('epoch')[-1].split('_')[0]), float(saved_model_path.split('valloss')[-1].split('.')[0])
        
            best_val_loss = {"epoch": load_epoch+1, "loss": load_loss}
        
            print ("Load model from %s" % saved_model_path)
            seq2seq.load_state_dict(torch.load(saved_model_path))
    
    for epoch in range(best_val_loss["epoch"], NUM_EPOCHS+1):
        start_time = timer()
        if not args.pytorch_seq2seq:
            train_loss = train_epoch(train_dataset, (encoder, decoder), (encoder_optimizer, decoder_optimizer, encoder_scheduler, decoder_scheduler), loss_fn, device)
        else:
            train_loss = train_epoch(train_dataset, seq2seq, (optimizer, scheduler), loss_fn, device)

        end_time = timer()
        
        with torch.no_grad():
            if not args.pytorch_seq2seq:
                val_loss = evaluate(val_dataloader, (encoder, decoder), loss_fn, device)
            else:
                val_loss = evaluate(val_dataloader, seq2seq, loss_fn, device)
        val_loss = np.mean(val_loss)
        
        if val_loss < best_val_loss["loss"]:

            best_val_loss["loss"] = val_loss
            best_val_loss["epoch"] = epoch
            
            if not args.pytorch_seq2seq:
                torch.save(encoder.state_dict(), 'trained_models/encoder_epoch%d_valloss%f.pt' % (epoch, val_loss))
                torch.save(decoder.state_dict(), 'trained_models/decoder_epoch%d_valloss%f.pt' % (epoch, val_loss))
            else:
                torch.save(seq2seq.state_dict(), 'trained_models/seq2seq_epoch%d_valloss%f.pt' % (epoch, val_loss))

        if epoch % 10 == 0:
            if not args.pytorch_seq2seq:
                torch.save(encoder.state_dict(), 'trained_models/encoder_epoch%d_valloss%f.pt' % (epoch, val_loss))
                torch.save(decoder.state_dict(), 'trained_models/decoder_epoch%d_valloss%f.pt' % (epoch, val_loss))
            else:
                torch.save(seq2seq.state_dict(), 'trained_models/seq2seq_epoch%d_valloss%f.pt' % (epoch, val_loss))
                
        print((f"Epoch: {epoch}, Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}", f"Epoch time = {(end_time - start_time):.3f}s"))
