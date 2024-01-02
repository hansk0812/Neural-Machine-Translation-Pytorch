import os
import glob
import numpy as np

import torch
from torch import nn

from dataset import vocab_transform, SRC_LANGUAGE, TGT_LANGUAGE
from dataset import BOS_IDX, EOS_IDX, PAD_IDX
from dataset import Multi30k, collate_fn

from dataset import text_transform, SRC_LANGUAGE, TGT_LANGUAGE

from model import Seq2SeqTransformer, DEVICE
from model import create_mask

from torch.utils.data import DataLoader

from timeit import default_timer as timer

torch.manual_seed(0)

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 160
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

def train_epoch(model, optimizer):
    model.train()
    losses = 0
    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for idx, (src, tgt) in enumerate(train_dataloader):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(list(train_dataloader))

def evaluate(model):
    model.eval()
    losses = 0

    val_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(val_dataloader))

if __name__ == "__main__":

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    NUM_EPOCHS = 18

    if not os.path.isdir('trained_models'):
        best_val_loss = {"epoch": 1, "loss": np.inf}
        os.mkdir("trained_models")
    else:
        saved_model_path = sorted(glob.glob('trained_models/*.pt'), key=lambda x: float(x.split('valloss')[-1].split('.')[0]))[0]
        load_epoch, load_loss = int(saved_model_path.split('epoch')[-1].split('_')[0]), float(saved_model_path.split('valloss')[-1].split('.')[0])
        best_val_loss = {"epoch": load_epoch+1, "loss": load_loss}
        
        print ("Load model from %s" % saved_model_path)
        transformer.load_state_dict(torch.load(saved_model_path))

    for epoch in range(best_val_loss["epoch"], NUM_EPOCHS+1):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer)
        end_time = timer()
        val_loss = evaluate(transformer)
        
        if val_loss < best_val_loss["loss"] and epoch > 10:

            best_val_loss["loss"] = val_loss
            best_val_loss["epoch"] = epoch

            torch.save(transformer.state_dict(), 'trained_models/epoch%d_valloss%f.pt' % (epoch, val_loss))
        
        if epoch % 10 == 0:
            torch.save(transformer.state_dict(), 'trained_models/epoch%d_valloss%f.pt' % (epoch, val_loss))

        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

