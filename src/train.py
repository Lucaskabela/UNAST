'''
Contains the code for training the encoder/decoders, including:
    - Cross model loss
    - Supervised loss
    - Denoising loss
    - Discriminator loss
'''
from utils import *
from preprocess import get_dataset, DataLoader, collate_fn_transformer
from module import TextPrenet, TextPostnet, RNNDecoder, RNNEncoder
from network import TextRNN, SpeechRNN, TextTransformer, SpeechTransformer, UNAST, Discriminator, LSTMDiscriminator
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
import audio_parameters as ap
import argparse
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from data import sequence_to_text
import datetime
import math
import io
import os
import sys

# DEVICE, WRITER are the only global variable

class BatchGetter():
    def __init__(self, args, supervised_dataset, unsupervised_dataset, full_dataset):
        self.batch_size = args.train_batch_size
        self.num_workers = args.num_workers

        self.supervised_dataloader = DataLoader(supervised_dataset,
            batch_size=self.batch_size, shuffle=True,
            collate_fn=collate_fn_transformer, drop_last=True,
            num_workers=self.num_workers, pin_memory=True)
        self.supervised_iter = iter(self.supervised_dataloader)

        self.unsupervised_dataloader = DataLoader(unsupervised_dataset,
            batch_size=self.batch_size, shuffle=True,
            collate_fn=collate_fn_transformer, drop_last=True,
            num_workers=self.num_workers, pin_memory=True)
        self.unsupervised_iter = iter(self.unsupervised_dataloader)

        if args.use_discriminator:
            self.discriminator_dataloader = DataLoader(full_dataset,
                batch_size=self.batch_size, shuffle=True,
                collate_fn=collate_fn_transformer, drop_last=True,
                num_workers=self.num_workers, pin_memory=True)
            self.discriminator_iter = iter(self.discriminator_dataloader)

    def get_supervised_batch(self):
        try:
            batch = self.supervised_iter.next()
        except StopIteration:
            self.supervised_iter = iter(self.supervised_dataloader)
            batch = self.supervised_iter.next()
        return batch

    def get_unsupervised_batch(self):
        try:
            batch = self.unsupervised_iter.next()
        except StopIteration:
            self.unsupervised_iter = iter(self.unsupervised_dataloader)
            batch = self.unsupervised_iter.next()
        return batch

    def get_discriminator_batch(self):
        try:
            batch = self.discriminator_iter.next()
        except StopIteration:
            self.discriminator_iter = iter(self.discriminator_dataloader)
            batch = self.discriminator_iter.next()
        return batch

def process_batch(batch):
    # Pos_text is unused so don't even bother loading it up
    text, mel, text_len, mel_len = batch

    # Detach gold labels stuff we use a lot to device
    gold_mel, gold_char = mel.detach(), text.detach()
    # stop label should be 1 for length, subtracting 1 for 0 based indexing
    with torch.no_grad():
        gold_stop = F.one_hot(mel_len - 1, mel.shape[1]).float().detach()

    # Send stuff we use alot to device this is character, mel, mel_input, and pos_mel
    text, mel, = text.to(DEVICE), mel.to(DEVICE)
    text_len, mel_len = text_len.to(DEVICE), mel_len.to(DEVICE)

    return (text, mel, text_len, mel_len), (gold_char, gold_mel, gold_stop)


#####----- LOSS FUNCTIONS -----#####

# Masked MSE LOSS
def masked_mse(gold_mel, pred_mel, mel_mask):
    diff2 = (torch.flatten(gold_mel) - torch.flatten(pred_mel)) ** 2.0 * torch.flatten(mel_mask)
    result = torch.sum(diff2) / torch.sum(mel_mask)
    return result

def text_loss(gold_char, text_pred, eos_weight=1.0):
    weight = torch.ones(text_pred.shape[1], device=DEVICE)
    weight[EOS_IDX] = eos_weight
    return F.cross_entropy(text_pred,
                           gold_char,
                           weight=weight,
                           ignore_index=PAD_IDX)

def speech_loss(gold_mel, stop_label, pred_mel, post_pred_mel, mel_len, stop_pred, eos_weight=1.0):
    # Apply length mask to pred_mel!
    mel_mask = sent_lens_to_mask(mel_len, pred_mel.shape[1]).unsqueeze(-1).repeat(1, 1, pred_mel.shape[2])
    pred_loss = masked_mse(gold_mel, pred_mel, mel_mask)
    post_pred_loss = masked_mse(gold_mel, post_pred_mel, mel_mask)
    stop_weight = torch.where(stop_label.eq(1),
                              torch.tensor(eos_weight).to(DEVICE),
                              torch.ones(1, device=DEVICE))
    stop_loss = F.binary_cross_entropy_with_logits(stop_pred.squeeze(-1), stop_label, pos_weight=stop_weight)
    return pred_loss + post_pred_loss + stop_loss

def cross_entropy(input, target, size_average=True):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean

    Examples::

        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)

        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    logsoftmax = nn.LogSoftmax(1)
    if size_average:
        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))

def discriminator_loss(output, target):
    return F.binary_cross_entropy_with_logits(output, target)

def discriminator_target(batch_size, target_type, smoothing=0.1):
    """
    Create the target labels with smoothing
    Params
        -batch_size : batch size
        -type : 'text' or 'speech'
        -smoothing : label smoothing factor
    Return
        [batch_size] tensor with target smoothed labels
    """
    target = torch.ones(batch_size).float()
    target -= smoothing
    if target_type == 'speech':
        target = 1 - target
    return target

def check_nan_loss(model, loss, loss_type, text_gold, text_pred, speech_gold, speech_pred, stop_gold, stop_pred):
    if torch.isnan(loss):
        print(f"Discovered NaN loss on {loss_type}!")
        if text_gold is not None:
            print("Text gold:")
            for idx in range(text_gold.shape[0]):
                print(f"{idx}. {sequence_to_text(text_gold[idx].detach().cpu().numpy())}")
        if text_pred is not None:
            print("Text pred:")
            for idx in range(text_pred.shape[0]):
                print(f"{idx}. {sequence_to_text(text_pred[idx].detach().cpu().numpy())}")
        torch.set_printoptions(edgeitems=1000, profile="full")
        if speech_gold is not None:
            print("Speech gold:")
            for idx in range(speech_gold.shape[0]):
                print(idx)
                print(speech_gold[idx])
        if speech_pred is not None:
            print("Speech pred:")
            for idx in range(speech_pred.shape[0]):
                print(idx)
                print(speech_pred[idx])
        if stop_gold is not None:
            print("Stop gold:")
            print(stop_gold)
        if stop_pred is not None:
            print("Stop pred:")
            print(stop_pred)
        print("=============== MODEL ===============")
        print(model)
        sys.exit("Loss is NaN")

#####----- Use these to run a task on a batch ----#####
def autoencoder_step(model, batch, args, use_dis_loss=False):
    """
    Compute and return the loss for autoencoders
    """
    x, y = batch
    text, mel, text_len, mel_len  = x
    gold_char, gold_mel, gold_stop = y

    if use_dis_loss:
        text_pred, t_hid = model.text_ae(text, text_len, ret_enc_hid=use_dis_loss)
        text_pred = text_pred.permute(0, 2, 1)
        pre_pred, post_pred, stop_pred, s_hid = model.speech_ae(mel, mel_len, ret_enc_hid=use_dis_loss)

        d_batch = discriminator_shuffle_batch(t_hid, text_len, s_hid, mel_len, args.model_type)
        d_ae_loss, _ = discriminator_hidden_to_loss(model, d_batch, freeze_discriminator=True)
    else:
        text_pred = model.text_ae(text, text_len).permute(0, 2, 1)
        pre_pred, post_pred, stop_pred = model.speech_ae(mel, mel_len)

    s_ae_loss = speech_loss(gold_mel.to(DEVICE), gold_stop.to(DEVICE), pre_pred, post_pred, mel_len, stop_pred, args.s_eos_weight)
    t_ae_loss = text_loss(gold_char.to(DEVICE), text_pred, args.t_eos_weight)

    # Check loss is not NaN
    check_nan_loss(model, t_ae_loss, "ae_text_loss", text, text_pred, mel, post_pred, gold_stop, stop_pred)
    check_nan_loss(model, s_ae_loss, "ae_speech_loss", text, text_pred, mel, post_pred, gold_stop, stop_pred)
    if use_dis_loss:
        check_nan_loss(model, d_ae_loss, "ae_dis_loss", text, text_pred, mel, post_pred, gold_stop, stop_pred)

    if use_dis_loss:
        return t_ae_loss, s_ae_loss, d_ae_loss
    return t_ae_loss, s_ae_loss

def supervised_step(model, batch, args, use_dis_loss=False):
    x, y = batch
    text, mel, text_len, mel_len  = x
    gold_char, gold_mel, gold_stop = y

    mel_aug = specaugment(mel, mel_len)
    if use_dis_loss:
        pre_pred, post_pred, stop_pred, stop_lens, t_hid = model.tts(text, text_len, mel, mel_len, ret_enc_hid=use_dis_loss)
        text_pred, s_hid = model.asr(text, text_len, mel_aug, mel_len, ret_enc_hid=use_dis_loss)
        text_pred = text_pred.permute(0, 2, 1)

        d_batch = discriminator_shuffle_batch(t_hid, text_len, s_hid, mel_len, args.model_type)
        d_sp_loss, _ = discriminator_hidden_to_loss(model, d_batch, freeze_discriminator=True)
    else:
        pre_pred, post_pred, stop_pred, stop_lens = model.tts(text, text_len, mel, mel_len)
        text_pred = model.asr(text, text_len, mel_aug, mel_len).permute(0, 2, 1)

    tts_loss = speech_loss(gold_mel.to(DEVICE), gold_stop.to(DEVICE), pre_pred, post_pred, mel_len, stop_pred, args.s_eos_weight)
    asr_loss = text_loss(gold_char.to(DEVICE), text_pred, args.t_eos_weight)

    # Check loss is not NaN
    check_nan_loss(model, asr_loss, "sp_asr_loss", None, text_pred, mel_aug, None, None, None)
    check_nan_loss(model, tts_loss, "sp_tts_loss", text, None, None, post_pred, None, stop_pred)
    if use_dis_loss:
        check_nan_loss(model, d_sp_loss, "sp_dis_loss", text, text_pred, mel_aug, post_pred, gold_stop, stop_pred)

    if use_dis_loss:
        return asr_loss, tts_loss, d_sp_loss
    return asr_loss, tts_loss

def crossmodel_step(model, batch, args, use_dis_loss=False):
    #NOTE: not sure if this will fail bc multiple grads on the model...
    x, y = batch
    text, mel, text_len,  mel_len  = x
    gold_char, gold_mel, gold_stop = y

    # Do speech!
    if use_dis_loss:
        pre_pred, post_pred, stop_pred, cm_t_hid, cm_t_len = model.cm_speech_in(mel, mel_len, ret_enc_hid=use_dis_loss)
    else:
        pre_pred, post_pred, stop_pred = model.cm_speech_in(mel, mel_len)
    s_cm_loss = speech_loss(gold_mel.to(DEVICE), gold_stop.to(DEVICE), pre_pred, post_pred, mel_len, stop_pred, args.s_eos_weight)

    # Now do text!
    if use_dis_loss:
        text_pred, cm_s_hid, cm_s_len = model.cm_text_in(text, text_len, ret_enc_hid=use_dis_loss)
        text_pred = text_pred.permute(0, 2, 1)
    else:
        text_pred = model.cm_text_in(text, text_len).permute(0, 2, 1)
    t_cm_loss = text_loss(gold_char.to(DEVICE), text_pred, args.t_eos_weight)

    if use_dis_loss:
        d_batch = discriminator_shuffle_batch(cm_t_hid, cm_t_len, cm_s_hid, cm_s_len, args.model_type)
        d_cm_loss, _ = discriminator_hidden_to_loss(model, d_batch, freeze_discriminator=True)

    # Check loss is not NaN
    check_nan_loss(model, t_cm_loss, "cm_text_loss", text, text_pred, mel, post_pred, gold_stop, stop_pred)
    check_nan_loss(model, s_cm_loss, "cm_speech_loss", text, text_pred, mel, post_pred, gold_stop, stop_pred)
    if use_dis_loss:
        check_nan_loss(model, d_cm_loss, "cm_dis_loss", text, text_pred, mel, post_pred, gold_stop, stop_pred)

    if use_dis_loss:
        return t_cm_loss, s_cm_loss, d_cm_loss
    return t_cm_loss, s_cm_loss

def discriminator_shuffle_batch(t_hid, t_hid_len, s_hid, s_hid_len, model_type, train_discriminator=False):
    if model_type == 'rnn':
        _, t_out = t_hid
        _, s_out = s_hid
    else:
        t_out = t_hid
        s_out = s_hid

    # Construct targets
    t_target = discriminator_target(t_out.shape[0], 'text').to(t_out)
    s_target = discriminator_target(s_out.shape[0], 'speech').to(s_out)

    # Pad both hidden states' sequence length with PAD_IDX to match dims
    t_seq_dim = t_out.shape[1]
    s_seq_dim = s_out.shape[1]
    d_hid = pad_sequence([t_out.permute(1, 0, 2), s_out.permute(1, 0, 2)], padding_value=PAD_IDX)

    # Remove catted padding sequence then recat on batch size
    d_hid = torch.cat(torch.unbind(d_hid, dim=1), dim=1).permute(1, 0, 2)

    # Concatenate
    d_len = torch.cat([t_hid_len, s_hid_len], dim=0)
    d_target = torch.cat([t_target, s_target], dim=0)
    if not train_discriminator:
        d_target = 1 - d_target

    # Shuffle
    indices = torch.randperm(d_hid.shape[0])
    d_hid = d_hid[indices]
    d_len = d_len[indices]
    d_target = d_target[indices]

    d_batch = (d_hid, d_len, d_target)
    return d_batch

def discriminator_hidden_to_loss(model, d_batch, freeze_discriminator=False):
    d_loss = 0
    if freeze_discriminator:
        with torch.no_grad():
            d_hid, d_len, d_target = d_batch
            d_out = model.discriminator(d_hid, d_len)
            d_loss += discriminator_loss(d_out, d_target)
    else:
        d_hid, d_len, d_target = d_batch
        d_out = model.discriminator(d_hid, d_len)
        d_loss += discriminator_loss(d_out, d_target)
    return d_loss, (d_out, d_target)

def discriminator_step(model, batch, args):
    x, _ = batch
    text, mel, text_len, mel_len = x

    # text and speech encoded representations
    with torch.no_grad():
        t_enc_out, _ = model.text_m.encode(text, text_len)
        s_enc_out, _ = model.speech_m.encode(mel, mel_len)

    # quick check to determine between rnn and transformer
    # eventually should be built into the RNN and Transformer Encoder classes
    d_batch = discriminator_shuffle_batch(t_enc_out, text_len, s_enc_out, mel_len, args.model_type, train_discriminator=True)
    d_loss, d_output = discriminator_hidden_to_loss(model, d_batch)

    # Check loss is not NaN
    check_nan_loss(model, d_loss, "dis_loss", text, None, mel, None, None, None)

    return d_loss, d_output


#####---- Use these to train on a task -----#####
def optimizer_step(model, optimizer, args):
    # Take a optimizer step!
    if args.grad_clip > 0.0:
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

def train_sp_step(losses, model, batch, step, accum_steps, args):
    batch = process_batch(batch)
    if args.use_discriminator:
        asr_loss, tts_loss, d_sp_loss = supervised_step(model, batch, args, args.use_discriminator)
        loss = tts_loss + asr_loss + d_sp_loss
    else:
        asr_loss, tts_loss = supervised_step(model, batch, args)
        loss = tts_loss + asr_loss
    loss = loss / accum_steps

    loss.backward()

    # Log losses
    losses['asr_'].append(asr_loss.detach().cpu().item())
    losses['tts_'].append(tts_loss.detach().cpu().item())
    if args.use_discriminator:
        losses['sp_d'].append(d_sp_loss.detach().cpu().item())

    # Log to tensorboard
    if WRITER:
        WRITER.add_scalar('train/sp_asr_loss', asr_loss.detach().cpu().item(), step)
        WRITER.add_scalar('train/sp_tts_loss', tts_loss.detach().cpu().item(), step)
        if args.use_discriminator:
            WRITER.add_scalar('train/sp_dis_loss', d_sp_loss.detach().cpu().item(), step)

    return loss

def train_ae_step(losses, model, batch, step, accum_steps, args):
    batch = process_batch(batch)
    if args.use_discriminator:
        t_ae_loss, s_ae_loss, d_ae_loss = autoencoder_step(model, batch, args, args.use_discriminator)
        loss = t_ae_loss + s_ae_loss + d_ae_loss
    else:
        t_ae_loss, s_ae_loss = autoencoder_step(model, batch, args)
        loss = t_ae_loss + s_ae_loss
    loss = loss / accum_steps
    loss.backward()

    # Log losses
    losses['t_ae'].append(t_ae_loss.detach().cpu().item())
    losses['s_ae'].append(s_ae_loss.detach().cpu().item())
    if args.use_discriminator:
        losses['d_ae'].append(d_ae_loss.detach().cpu().item())

    # Log to tensorboard
    if WRITER:
        WRITER.add_scalar('train/ae_t_loss', t_ae_loss.detach().cpu().item(), step)
        WRITER.add_scalar('train/ae_s_loss', s_ae_loss.detach().cpu().item(), step)
        if args.use_discriminator:
            WRITER.add_scalar('train/ae_d_loss', d_ae_loss.detach().cpu().item(), step)

    return loss

def train_cm_step(losses, model, batch, step, accum_steps, args):
    # NOTE: do not use cross_model here bc need to take optimizer step inbetween
    batch = process_batch(batch)

    if args.use_discriminator:
        t_cm_loss, s_cm_loss, d_cm_loss = crossmodel_step(model, batch, args, args.use_discriminator)
        loss = s_cm_loss + t_cm_loss + d_cm_loss
    else:
        t_cm_loss, s_cm_loss = crossmodel_step(model, batch, args)
        loss = s_cm_loss + t_cm_loss
    loss = loss / accum_steps
    loss.backward()

    # Log losses
    losses['s_cm'].append(s_cm_loss.detach().cpu().item())
    losses['t_cm'].append(t_cm_loss.detach().cpu().item())
    if args.use_discriminator:
        losses['d_cm'].append(d_cm_loss.detach().cpu().item())

    # Log to tensorboard
    if WRITER:
        WRITER.add_scalar('train/cm_t_loss', t_cm_loss.detach().cpu().item(), step)
        WRITER.add_scalar('train/cm_s_loss', s_cm_loss.detach().cpu().item(), step)
        if args.use_discriminator:
            WRITER.add_scalar('train/cm_d_loss', d_cm_loss.detach().cpu().item(), step)

    return loss

def train_discriminator_step(losses, model, batch, step, accum_steps, args, log_out_to_tb=False):
    batch = process_batch(batch)
    d_loss, d_output = discriminator_step(model, batch, args)

    loss = d_loss / accum_steps
    loss.backward()

    # Log losses
    losses['d'].append(d_loss.detach().cpu().item())

    # Log to tensorboard
    if WRITER:
        WRITER.add_scalar('train/dis_loss', d_loss.detach().cpu().item(), step)
        if log_out_to_tb:
            d_out, d_target = d_output
            log_tb_discrim_out(d_out, d_target, step)

    return loss

def freeze_model_parameters(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_model_parameters(model):
    for param in model.parameters():
        param.requires_grad = True

#####----- Train and evaluate -----#####
def evaluate(model, valid_dataloader, step, args):
    """
        Expect validation set to have paired speech & text!
        Primary evaluation is PER - can gauge training by the other losses
        We return on 6 other metrics:  autoencoder text loss, autoencoder speech loss,
            ASR loss, and TTS loss, cross model text and cross model speech loss.
    """
    model.eval()
    with torch.no_grad():
        losses = defaultdict(list)
        per, n_iters = 0, 0

        for batch in valid_dataloader:
            batch = process_batch(batch)
            x, _ = batch
            text, mel, text_len, mel_len = x

            if args.use_discriminator:
                t_ae_loss, s_ae_loss, d_ae_loss = autoencoder_step(model, batch, args, args.use_discriminator)
            else:
                t_ae_loss, s_ae_loss = autoencoder_step(model, batch, args)
            losses['t_ae'].append(t_ae_loss.detach().item())
            losses['s_ae'].append(s_ae_loss.detach().item())
            if args.use_discriminator:
                losses['d_ae'].append(d_ae_loss.detach().item())

            if args.use_discriminator:
                asr_loss, tts_loss, d_sp_loss = supervised_step(model, batch, args, args.use_discriminator)
            else:
                asr_loss, tts_loss = supervised_step(model, batch, args)
            losses['asr'].append(asr_loss.detach().item())
            losses['tts'].append(tts_loss.detach().item())
            if args.use_discriminator:
                losses['d_sp'].append(d_sp_loss.detach().item())

            if args.use_discriminator:
                t_cm_loss, s_cm_loss, d_cm_loss = crossmodel_step(model, batch, args, args.use_discriminator)
            else:
                t_cm_loss, s_cm_loss = crossmodel_step(model, batch, args)
            losses['s_cm'].append(s_cm_loss.detach().item())
            losses['t_cm'].append(t_cm_loss.detach().item())
            if args.use_discriminator:
                losses['d_cm'].append(d_cm_loss.detach().item())

            if args.use_discriminator:
                d_loss, d_output = discriminator_step(model, batch, args)
                losses['dis'].append(d_loss.detach().item())

            text_pred, text_pred_len = model.asr(None, None, mel, mel_len, infer=True)
            per += compute_per(text, text_pred.squeeze(-1), text_len, text_pred_len)
            n_iters += 1

        # TODO: evaluate speech inference somehow?
        compare_outputs(text[-1][:], text_pred[-1][:], text_len[-1], text_pred_len[-1])
        if args.use_discriminator:
            d_out, d_target = d_output
            log_tb_discrim_out(d_out, d_target, step, "eval")

    return per/n_iters, losses

def train(args):
    set_seed(args.seed)
    supervised_train_dataset, unsupervised_train_dataset, val_dataset, full_train_dataset = \
        initialize_datasets(args)
    valid_dataloader = DataLoader(val_dataset,
            batch_size=args.eval_batch_size, shuffle=True,
            collate_fn=collate_fn_transformer, drop_last=True,
            num_workers=args.num_workers, pin_memory=True)
    batch_getter = BatchGetter(args, supervised_train_dataset, unsupervised_train_dataset, full_train_dataset)

    # Define epoch_steps, 1 step does all 4 tasks
    if args.epoch_steps <= 0:
        # Define an epoch to be one pass through the discriminator's train dataset
        # This makes it so the total number of steps is the same regardless # of whether we train the discriminator or not
        total_batches_full_dataset = math.ceil(len(full_train_dataset) / args.batch_size)
        args.epoch_steps = math.ceil(total_batches_full_dataset / args.d_steps)
    epoch_steps = args.epoch_steps

    s_epoch, best, model, optimizer, scheduler = initialize_model(args)

    print("Training model with {} parameters".format(model.num_params()))
    per, eval_losses = evaluate(model, valid_dataloader, -1, args)
    log_loss_metrics(eval_losses, -1, eval=True)

    max_obj_steps = max(args.ae_steps, args.cm_steps, args.sp_steps)
    if args.use_discriminator:
        max_obj_steps = max(max_obj_steps, args.d_steps)
    accum_steps = args.ae_steps + args.cm_steps + args.sp_steps

    for epoch in range(s_epoch, args.epochs):
        losses = defaultdict(list)

        bar = tqdm(range(0, epoch_steps))
        bar.set_description("Epoch {}".format(epoch))
        for s in bar:
            model.train()

            # if args.use_discriminator:
            #     # need to freeze the disciminator first
            #     freeze_model_parameters(model.discriminator)
            #     unfreeze_model_parameters(model.text_m)
            #     unfreeze_model_parameters(model.speech_m)

            # DENOISING AUTO ENCODER
            for si in range(0, args.ae_steps):
                batch = batch_getter.get_unsupervised_batch()
                step = epoch*epoch_steps*max_obj_steps + s*max_obj_steps + si
                train_ae_step(losses, model, batch, step, accum_steps, args)

            # CM TTS/ASR
            for si in range(0, args.cm_steps):
                batch = batch_getter.get_unsupervised_batch()
                step = epoch*epoch_steps*max_obj_steps + s*max_obj_steps + si
                train_cm_step(losses, model, batch, step, accum_steps, args)

            # SUPERVISED
            for si in range(0, args.sp_steps):
                batch = batch_getter.get_supervised_batch()
                step = epoch*epoch_steps*max_obj_steps + s*max_obj_steps + si
                train_sp_step(losses, model, batch, step, accum_steps, args)

            # Gradients have accumulated - lets back prop and free memory
            optimizer_step(model, optimizer, args)

            # DISCRIMINATOR
            if args.use_discriminator:
                # unfreeze_model_parameters(model.discriminator)
                # freeze_model_parameters(model.text_m)
                # freeze_model_parameters(model.speech_m)
                for si in range(0, args.d_steps):
                    batch = batch_getter.get_discriminator_batch()
                    step = epoch*epoch_steps*max_obj_steps + s*max_obj_steps + si
                    log_out_to_tb = si == 0 and (s + 1) % args.tb_example_step == 0
                    train_discriminator_step(losses, model, batch, step, args.d_steps, args, log_out_to_tb)
                optimizer_step(model, optimizer, args)

            # Monitor certain information
            if WRITER:
                # Learning rate
                step = epoch*epoch_steps*max_obj_steps + s*max_obj_steps
                WRITER.add_scalar("misc/learning_rate", optimizer.param_groups[0]['lr'], step)

                # Model weights
                step = epoch*epoch_steps*max_obj_steps + (s + 1)*max_obj_steps - 1
                sum_params = 0
                for param in model.parameters():
                    sum_params += torch.sum(param).abs().item()
                WRITER.add_scalar("misc/model_params_weight", sum_params, step)

            # LR scheduler step to change LR
            if scheduler is not None:
                scheduler.step()

            # Log train example to tensorboard
            if WRITER and (s + 1) % args.tb_example_step == 0:
                idx = np.random.randint(0, len(supervised_train_dataset))
                ex = supervised_train_dataset[idx]
                step = epoch*epoch_steps*max_obj_steps + (s + 1)*max_obj_steps - 1
                log_tb_example(model, ex, step)

        # Eval and save
        step = (epoch + 1)*epoch_steps*max_obj_steps - 1
        per, eval_losses = evaluate(model, valid_dataloader, step, args)
        log_loss_metrics(losses, epoch)
        log_loss_metrics(eval_losses, epoch, eval=True)

        # Log eval example to tensorboard
        if WRITER:
            idx = np.random.randint(0, len(val_dataset))
            ex = val_dataset[idx]
            log_tb_example(model, ex, step, "eval")
            for key_, loss in eval_losses.items():
                WRITER.add_scalar(f"eval/{key_}_loss", np.mean(loss), step)
            WRITER.add_scalar(f"eval/per", per, step)

        model.teacher.step()
        print("Eval_ epoch {:-3d} PER {:0.3f}\%".format(epoch, per*100))
        save_ckp(epoch, per, model, optimizer, per < best, args.checkpoint_path)
        if args.save_every is not None and (epoch + 1) % args.save_every == 0:
            save_ckp(epoch, per, model, optimizer, per < best, args.checkpoint_path, epoch_save=True)
        if per < best:
            print("\t Best score - saving model!")
            best = per
    model.eval()
    return model


def log_tb_example(model, ex, step, name="train"):
    """
    Log example of model output to tensorboard
    params
        - model: the ASR/TTS model
        - ex: an input example from LJSpeech
        - step: tensorboard global step
    """
    model.eval()
    with torch.no_grad():
        if WRITER:
            # Convert input to tensor
            ex_mel = torch.tensor(ex["mel"]).unsqueeze(0)
            ex_text = torch.LongTensor(ex["text"]).unsqueeze(0)
            ex_text_len = torch.as_tensor(ex["text_length"], dtype=torch.long).unsqueeze(0)
            ex_mel_len = torch.as_tensor(ex["mel_length"], dtype=torch.long).unsqueeze(0)

            # Get output
            text_pred, text_pred_len = model.asr(None, None, ex_mel.to(DEVICE), ex_mel_len.to(DEVICE), infer=True)
            text_pred = text_pred.squeeze(dim=0).detach().cpu().numpy()
            text_pred_len = text_pred_len.squeeze(dim=0).detach().cpu().item()
            _, speech_pred, _, speech_pred_len = model.tts(ex_text.to(DEVICE), ex_text_len.to(DEVICE), None, None, infer=True)
            speech_pred = speech_pred.squeeze(dim=0).detach().cpu().numpy()
            speech_pred_len = speech_pred_len.squeeze(dim=0).detach().cpu().item()

            WRITER.add_text(f"{name}/text_gold", sequence_to_text(ex["text"][:ex_text_len]), step)
            WRITER.add_text(f"{name}/text_pred", sequence_to_text(text_pred[:text_pred_len]), step)
            WRITER.add_image(f"{name}/speech_gold", np.flip(ex["mel"][:ex_mel_len].transpose(), axis=0), step, dataformats="HW")
            WRITER.add_image(f"{name}/speech_pred", np.flip(speech_pred[:speech_pred_len].transpose(), axis=0), step, dataformats="HW")


def log_tb_discrim_out(d_out, d_target, step, name="train"):
    """
    Log example of discriminator output to tensorboard
    params
        - d_out: discriminator output
        - d_target: discriminator target
        - step: tensorboard global step
    """
    if WRITER:
        d_out = torch.sigmoid(d_out)

        batch = d_out.shape[0]
        fig, ax = plt.subplots(figsize=(batch // 2, 3))

        # plot bars
        ind = np.arange(batch)
        width = 0.2
        ax.bar(ind, d_out.detach().cpu().numpy(), width, label="pred")
        ax.bar(ind + width, d_target.detach().cpu().numpy(), width, label="gold")
        ax.set_xticks(ind + width / 2)
        ax.set_xticklabels(ind)
        ax.legend()
        fig.tight_layout()

        WRITER.add_figure(f"{name}/discrim_output", fig, step)

def log_loss_metrics(losses, epoch, eval=False):
    kind = "Train"
    if eval:
        kind = "Eval_"

    out_str = "{} epoch {:-3d} \t".format(kind, epoch)
    for key_, loss in sorted(losses.items()):
        out_str += "{} loss =  {:0.3f} \t".format(key_, np.mean(loss))
    print(out_str)


def train_text_auto(args):
    ''' Purely for testing purposes'''
    set_seed(args.seed)

    # TODO: Replace get_dataset() with getting train/valid/test split
    dataset = get_dataset('unlabeled_train.csv')

    #NOTE: Subset for prototyping
    # dataset = torch.utils.data.Subset(dataset, range(1000))

    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn_transformer, drop_last=True, num_workers=16)
    train_log, valid_log = None, None
    model = TextRNN(args).to(DEVICE)
    print("Sent model to", DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,  weight_decay=1e-5)

    global_step = 0

    for epoch in range(args.epochs):
        losses = []
        for data in dataloader:
            character, mel, mel_input, pos_text, pos_mel, text_len = data
            character = character.to(DEVICE)
            pred = model.forward(character).permute(0, 2, 1)

            char_ = character
            loss = F.cross_entropy(pred, char_, ignore_index=PAD_IDX)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(loss.detach().item())
            if train_log is not None:
                train_log.add_scalar("loss", losses[-1], global_step)
            global_step += 1
        avg_l = np.mean(losses)
        print("epoch %-3d \t loss = %0.3f \t" % (epoch, avg_l))
    return model

def train_speech_auto(args):
    ''' Purely for testing purposes'''
    set_seed(args.seed)

    # TODO: Replace get_dataset() with getting train/valid/test split
    dataset = get_dataset('unlabeled_train.csv')
    #NOTE: Subset for prototyping
    # dataset = torch.utils.data.Subset(dataset, range(1000))
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn_transformer, drop_last=True, num_workers=16)
    train_log, valid_log = None, None
    model = SpeechRNN(args).to(DEVICE)
    print("Sent model to", DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,  weight_decay=1e-5)

    global_step = 0

    for epoch in range(args.epochs):
        losses = []
        for data in dataloader:
            character, mel, mel_input, pos_text, pos_mel, text_len = data
            mel_input, mel = mel_input.to(DEVICE), mel.to(DEVICE)
            pred, stop_pred = model.forward(mel, mel_input)


            pred_loss = F.mse_loss(pred, mel)
            # Should be [batch_size x seq_length] for stop

            # TODO: return actual lengths, not just computed off padding
            # currently, find first nonzero (so pad_idx) in pos_mel, or set to length
            end_mask_max, end_mask_idx = torch.max((pos_mel == PAD_IDX), dim=1)
            end_mask_idx[end_mask_max == 0] = pos_mel.shape[1] - 1
            stop_label = F.one_hot(end_mask_idx, pos_mel.shape[1]).float()
            stop_loss = F.binary_cross_entropy_with_logits(stop_pred.squeeze(), stop_label)

            loss = pred_loss + stop_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(loss.detach().item())
            if train_log is not None:
                train_log.add_scalar("loss", losses[-1], global_step)
            global_step += 1
        avg_l = np.mean(losses)
        print("epoch %-3d \t loss = %0.3f \t" % (epoch, avg_l))
    return model


#####----- Model, optimizer, scheduler, dataset initializations -----#####
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Copied and modified from https://huggingface.co/transformers/_modules/transformers/optimization.html#get_constant_schedule_with_warmup

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_transformer_paper_schedule(optimizer, num_warmup_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that follows from the paper "Attention is all you need",
    a linear warmup period followed by decrease that's inversely proportionl to square root of step number.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / max(1.0, float(num_warmup_steps)**1.5)
        return 1.0 / max(1.0, float(current_step)**0.5)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def initialize_model(args):
    """
        Using args, initialize starting epoch, best per, model, optimizer
    """
    text_m, speech_m, discriminator, teacher = None, None, None, get_teacher_ratio(args)
    if args.model_type == 'rnn':
        text_m = TextRNN(args)
        speech_m = SpeechRNN(args)
    elif args.model_type == 'transformer':
        text_m = TextTransformer(args)
        speech_m = SpeechTransformer(args)

    if args.use_discriminator:
        discriminator_in_dim = args.hidden * 2 if args.model_type == 'rnn' else args.hidden
        discriminator = LSTMDiscriminator(discriminator_in_dim, args.disc_hid, bidirectional=args.disc_bidirectional, num_layers=args.disc_num_layers)
    model = UNAST(text_m, speech_m, discriminator, teacher).to(DEVICE)

    # initialize optimizer
    optimizer = None
    if args.optim_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # continue training if needed
    s_epoch, best = 0, 300
    if args.load_path is not None:
        if os.path.isfile(args.load_path):
            s_epoch, best, model, optimizer = load_ckp(args.load_path, model, optimizer)
        else:
            print(f"[WARN] Could not find checkpoint '{args.load_path}'.")
            print(f"[WARN] Training from initial model...")

    # initialize scheduler
    scheduler = None
    if args.sched_type == 'multistep':
        milestones = [i * args.epoch_steps for i in args.lr_milestones]
        last_step = s_epoch * args.epoch_steps
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=args.lr_gamma, last_epoch=last_step-1)
    elif args.sched_type == 'linear':
        num_training_steps = args.epochs * args.epoch_steps
        last_step = s_epoch * args.epoch_steps
        scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, num_training_steps, last_step-1)
    elif args.sched_type == 'transformer':
        last_step = s_epoch * args.epoch_steps
        scheduler = get_transformer_paper_schedule(optimizer, args.warmup_steps, last_step-1)

    model.teacher.iter = s_epoch
    return s_epoch, best, model, optimizer, scheduler

def initialize_datasets(args):
    # TODO: replace these if we want
    print("#### Getting Dataset ####")
    supervised_train_dataset = get_dataset('labeled_train.csv')
    unsupervised_train_dataset = get_dataset('unlabeled_train.csv')
    val_dataset = get_dataset('val.csv')
    full_train_dataset = get_dataset('full_train.csv')

    # TODO: remove the subsets used for experimentation
    #supervised_train_dataset = torch.utils.data.Subset(supervised_train_dataset, range(100))
    #unsupervised_train_dataset = torch.utils.data.Subset(unsupervised_train_dataset, range(100))
    #val_dataset = torch.utils.data.Subset(val_dataset, range(200))
    #full_train_dataset = torch.utils.data.Subset(full_train_dataset, range(100))

    return supervised_train_dataset, unsupervised_train_dataset, val_dataset, full_train_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='JSON config files')
    args = parse_with_config(parser)

    global DEVICE
    global WRITER
    DEVICE = init_device(args)
    print(f"[{datetime.datetime.now()}] Device: {DEVICE}")

    if args.tb_log_path:
        WRITER = SummaryWriter(log_dir=args.tb_log_path, flush_secs=60)
        WRITER.add_text("params", str(vars(args)), 0)

    train(args)

    if WRITER:
        WRITER.close()
