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
from network import TextRNN, SpeechRNN, TextTransformer, SpeechTransformer, UNAST, Discriminator
from tqdm import tqdm
import audio_parameters as ap
import argparse
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from collections import defaultdict
from data import sequence_to_text
import math

# DEVICE is only global variable
def adjust_learning_rate(optimizer, e_in, step_num, warmup_step=4000):
    lr = e_in**-0.5 * min(step_num * warmup_step**-1.5, step_num**-0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class BatchGetter():
    def __init__(self, args, supervised_dataset, unsupervised_dataset, full_dataset):
        self.batch_size = args.batch_size
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

# TODO: add weights for the losses in args?
def text_loss(gold_char, text_pred):
    return F.cross_entropy(text_pred, gold_char, ignore_index=PAD_IDX)

def speech_loss(gold_mel, stop_label, pred_mel, post_pred_mel, mel_len, stop_pred):
    # Apply length mask to pred_mel!
    mel_mask = sent_lens_to_mask(mel_len, pred_mel.shape[1]).unsqueeze(-1).repeat(1, 1, pred_mel.shape[2])
    pred_loss = masked_mse(gold_mel, pred_mel, mel_mask)
    post_pred_loss = masked_mse(gold_mel, post_pred_mel, mel_mask)
    stop_loss = F.binary_cross_entropy_with_logits(stop_pred, stop_label)
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
    return cross_entropy(output, target)

def discriminator_target(output, target_type, smoothing=0.1):
    """
    Create the target labels with smoothing
    Params
        -output : [batch_size x 2]
        -type : 'text' or 'speech'
        -smoothing : label smoothing factor
    Return
        [batch_size x 2] tensor with target smoothed labels
    """
    target = torch.zeros_like(output)
    if target_type == 'text':
        smoothed_target_label = [1 - smoothing, smoothing]
    else:
        # target_type == 'speech' implicitly
        smoothed_target_label = [smoothing, 1 - smoothing]
    smoothed_target_label = torch.as_tensor(smoothed_target_label).to(target)
    target[:,] = smoothed_target_label
    return target



#####----- Use these to run a task on a batch ----#####
def autoencoder_step(model, batch, use_dis_loss=False):
    """
    Compute and return the loss for autoencoders
    """
    x, y = batch
    text, mel, text_len, mel_len  = x
    gold_char, gold_mel, gold_stop = y

    if use_dis_loss:
        text_pred, t_hid = model.text_ae(text, text_len, ret_enc_hid=use_dis_loss)
        text_pred = text_pred.permute(0, 2, 1)
        t_d_loss = discriminator_hidden_to_loss(model, t_hid, 'speech', freeze_discriminator=True)

        pre_pred, post_pred, stop_pred, s_hid = model.speech_ae(mel, mel_len, ret_enc_hid=use_dis_loss)
        s_d_loss = discriminator_hidden_to_loss(model, s_hid, 'text', freeze_discriminator=True)
    else:
        text_pred = model.text_ae(text, text_len).permute(0, 2, 1)
        pre_pred, post_pred, stop_pred = model.speech_ae(mel, mel_len)

    s_ae_loss = speech_loss(gold_mel.to(DEVICE), gold_stop.to(DEVICE), pre_pred, post_pred, mel_len, stop_pred)
    t_ae_loss = text_loss(gold_char.to(DEVICE), text_pred)
    if use_dis_loss:
        return t_ae_loss, s_ae_loss, t_d_loss, s_d_loss
    return t_ae_loss, s_ae_loss

def supervised_step(model, batch, use_dis_loss=False):
    x, y = batch
    text, mel, text_len, mel_len  = x
    gold_char, gold_mel, gold_stop = y

    mel_aug = specaugment(mel, mel_len)
    if use_dis_loss:
        pre_pred, post_pred, stop_pred, t_hid = model.tts(text, text_len, mel, mel_len, ret_enc_hid=use_dis_loss)
        t_d_loss = discriminator_hidden_to_loss(model, t_hid, 'speech', freeze_discriminator=True)

        text_pred, s_hid = model.asr(text, text_len, mel_aug, mel_len, ret_enc_hid=use_dis_loss)
        text_pred = text_pred.permute(0, 2, 1)
        s_d_loss = discriminator_hidden_to_loss(model, s_hid, 'text', freeze_discriminator=True)
    else:
        pre_pred, post_pred, stop_pred = model.tts(text, text_len, mel, mel_len)
        text_pred = model.asr(text, text_len, mel_aug, mel_len).permute(0, 2, 1)

    tts_loss = speech_loss(gold_mel.to(DEVICE), gold_stop.to(DEVICE), pre_pred, post_pred, mel_len, stop_pred)
    asr_loss = text_loss(gold_char.to(DEVICE), text_pred)
    if use_dis_loss:
        return asr_loss, tts_loss, t_d_loss, s_d_loss
    return asr_loss, tts_loss

def crossmodel_step(model, batch, use_dis_loss=False):
    #NOTE: not sure if this will fail bc multiple grads on the model...
    x, y = batch
    text, mel, text_len,  mel_len  = x
    gold_char, gold_mel, gold_stop = y

    # Do speech!
    if use_dis_loss:
        pre_pred, post_pred, stop_pred, cm_t_hid = model.cm_speech_in(mel, mel_len, ret_enc_hid=use_dis_loss)
        cm_t_d_loss = discriminator_hidden_to_loss(model, cm_t_hid, 'speech', freeze_discriminator=True)
    else:
        pre_pred, post_pred, stop_pred = model.cm_speech_in(mel, mel_len)
    s_cm_loss = speech_loss(gold_mel.to(DEVICE), gold_stop.to(DEVICE), pre_pred, post_pred, mel_len, stop_pred)

    # Now do text!
    if use_dis_loss:
        text_pred, cm_s_hid = model.cm_text_in(text, text_len, ret_enc_hid=args.use_discriminator)
        text_pred = text_pred.permute(0, 2, 1)
        cm_s_d_loss = discriminator_hidden_to_loss(model, cm_s_hid, 'text', freeze_discriminator=True)
    else:
        text_pred = model.cm_text_in(text, text_len).permute(0, 2, 1)
    t_cm_loss = text_loss(gold_char.to(DEVICE), text_pred)

    if use_dis_loss:
        return t_cm_loss, s_cm_loss, cm_t_d_loss, cm_s_d_loss
    return t_cm_loss, s_cm_loss

def discriminator_hidden_to_loss(model, hid, target_type, freeze_discriminator= False):
    d_in = hid[-1]
    if freeze_discriminator:
        with torch.no_grad():
            d_out = model.discriminator(d_in)
    else:
        d_out = model.discriminator(d_in)
    target = discriminator_target(d_out, target_type)
    d_loss = discriminator_loss(d_out, target)
    return d_loss

def discriminator_step(model, batch):
    x, _ = process_batch(batch)
    text, mel, text_len, mel_len = x

    # text
    with torch.no_grad():
        t_enc_out, _ = model.text_m.encode(text, text_len)
    # quick check to determine between rnn and transformer
    # eventually should be built into the RNN and Transformer Encoder classes
    if len(t_enc_out) == 2:
        t_hid = t_enc_out[0]
    else:
        t_hid = t_enc_out
    t_d_loss = discriminator_hidden_to_loss(model, t_hid, 'text')

    # speech
    with torch.no_grad():
        s_enc_out, _ = model.speech_m.encode(mel, mel_len)
    if len(s_enc_out) == 2:
        s_hid = s_enc_out[0]
    else:
        s_hid = s_enc_out
    s_d_loss = discriminator_hidden_to_loss(model, s_hid, 'speech')
    return t_d_loss, s_d_loss


#####---- Use these to train on a task -----#####
def optimizer_step(model, optimizer, args):

    # Take a optimizer step!
    if args.grad_clip > 0.0:
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

def train_sp_step(losses, model, batch, accum_steps, use_discriminator):
    batch = process_batch(batch)
    if use_discriminator:
        asr_loss, tts_loss, t_d_loss, s_d_loss = supervised_step(model, batch, use_discriminator)
        loss = tts_loss + asr_loss + t_d_loss + s_d_loss
    else:
        asr_loss, tts_loss = supervised_step(model, batch)
        loss = tts_loss + asr_loss
    loss = loss / accum_steps

    loss.backward()

    # Log losses
    losses['asr_'].append(asr_loss.detach().cpu().item())
    losses['tts_'].append(tts_loss.detach().cpu().item())
    if use_discriminator:
        losses['t_sp_d'].append(t_d_loss.detach().cpu().item())
        losses['s_sp_d'].append(s_d_loss.detach().cpu().item())
    return loss

def train_ae_step(losses, model, batch, accum_steps, use_discriminator):
    batch = process_batch(batch)
    if use_discriminator:
        t_ae_loss, s_ae_loss, t_d_loss, s_d_loss = autoencoder_step(model, batch, use_discriminator)
        loss = t_ae_loss + s_ae_loss + t_d_loss + s_d_loss
    else:
        t_ae_loss, s_ae_loss = autoencoder_step(model, batch)
        loss = t_ae_loss + s_ae_loss
    loss = loss / accum_steps
    loss.backward()

    # Log losses
    losses['t_ae'].append(t_ae_loss.detach().cpu().item())
    losses['s_ae'].append(s_ae_loss.detach().cpu().item())
    if use_discriminator:
        losses['t_ae_d'].append(t_d_loss.detach().cpu().item())
        losses['s_ae_d'].append(s_d_loss.detach().cpu().item())
    return loss

def train_cm_step(losses, model, batch, accum_steps, use_discriminator):
    # NOTE: do not use cross_model here bc need to take optimizer step inbetween
    batch = process_batch(batch)

    if use_discriminator:
        t_cm_loss, s_cm_loss, cm_t_d_loss, cm_s_d_loss = crossmodel_step(model, batch, use_discriminator)
        loss = s_cm_loss + t_cm_loss + cm_t_d_loss + cm_s_d_loss
    else:
        t_cm_loss, s_cm_loss = crossmodel_step(model, batch)
        loss = s_cm_loss + t_cm_loss
    loss = loss / accum_steps
    loss.backward()

    # Log losses
    losses['s_cm'].append(s_cm_loss.detach().cpu().item())
    losses['t_cm'].append(t_cm_loss.detach().cpu().item())
    if args.use_discriminator:
        losses['cm_s_cm_d'].append(cm_s_d_loss.detach().cpu().item())
        losses['cm_t_cm_d'].append(cm_t_d_loss.detach().cpu().item())
    return loss

def train_discriminator_step(losses, model, batch, accum_steps):
    t_d_loss, s_d_loss = discriminator_step(model, batch)

    loss = (t_d_loss + s_d_loss) / accum_steps
    loss.backward()

    losses['t_d'].append(t_d_loss.detach().cpu().item())
    losses['s_d'].append(s_d_loss.detach().cpu().item())
    return loss

def freeze_model_parameters(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_model_parameters(model):
    for param in model.parameters():
        param.requires_grad = True

#####----- Train and evaluate -----#####
def evaluate(model, valid_dataloader):
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

            t_ae_loss, s_ae_loss = autoencoder_step(model, batch)
            losses['t_ae'].append(t_ae_loss.detach().item())
            losses['s_ae'].append(s_ae_loss.detach().item())

            asr_loss, tts_loss = supervised_step(model, batch)
            losses['asr_'].append(asr_loss.detach().item())
            losses['tts_'].append(tts_loss.detach().item())

            t_cm_loss, s_cm_loss = crossmodel_step(model, batch)
            losses['s_cm'].append(s_cm_loss.detach().item())
            losses['t_cm'].append(t_cm_loss.detach().item())

            text_pred, text_pred_len = model.asr(None, None, mel, mel_len, infer=True)
            per += compute_per(text, text_pred.squeeze(), text_len, text_pred_len)
            n_iters += 1

    # TODO: evaluate speech inference somehow?
    compare_outputs(text[-1][:], text_pred[-1][:], text_len[-1], text_pred_len[-1])
    return per/n_iters, losses

def train(args):
    set_seed(args.seed)
    supervised_train_dataset, unsupervised_train_dataset, val_dataset, full_train_dataset = \
        initialize_datasets(args)
    valid_dataloader = DataLoader(val_dataset,
            batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_fn_transformer, drop_last=True,
            num_workers=args.num_workers, pin_memory=True)
    batch_getter = BatchGetter(args, supervised_train_dataset, unsupervised_train_dataset, full_train_dataset)

    s_epoch, best, model, optimizer = initialize_model(args)
    milestones = [i - s_epoch for i in args.lr_milestones if (i - s_epoch > 0)]
    sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=args.lr_gamma)

    print("Training model with {} parameters".format(model.num_params()))
    per, eval_losses = evaluate(model, valid_dataloader)
    log_loss_metrics(eval_losses, -1, eval=True)

    accum_steps = args.ae_steps + args.cm_steps + args.sp_steps
    # one step is doing all 4 training tasks
    if args.epoch_steps >= 0:
        epoch_steps = args.epoch_steps
    else:
        # Define an epoch to be one pass through the discriminator's train
        # dataset
        # This makes it so the total number of steps is the same regardless # of whether we train the discriminator or not
        total_batches_full_dataset = math.ceil(len(full_train_dataset) / args.batch_size)
        epoch_steps = math.ceil(total_batches_full_dataset / args.d_steps)
    total_step = s_epoch * epoch_steps

    for epoch in range(s_epoch, args.epochs):
        model.train()
        losses = defaultdict(list)

        bar = tqdm(range(0, epoch_steps))
        bar.set_description("Epoch {}".format(epoch))
        for _ in bar:
            # if args.use_discriminator:
            #     # need to freeze the disciminator first
            #     freeze_model_parameters(model.discriminator)
            #     unfreeze_model_parameters(model.text_m)
            #     unfreeze_model_parameters(model.speech_m)

            # DENOISING AUTO ENCODER
            for _ in range(0, args.ae_steps):
                batch = batch_getter.get_unsupervised_batch()
                train_ae_step(losses, model, batch, accum_steps, args.use_discriminator)

            # CM TTS/ASR
            for _ in range(0, args.cm_steps):
                batch = batch_getter.get_unsupervised_batch()
                train_cm_step(losses, model, batch, accum_steps, args.use_discriminator)

            # SUPERVISED
            for _ in range(0, args.sp_steps):
                batch = batch_getter.get_supervised_batch()
                train_sp_step(losses, model, batch, accum_steps, args.use_discriminator)

            # Gradients have accumulated - lets back prop and free memory
            optimizer_step(model, optimizer, args)
            # DISCRIMINATOR
            if args.use_discriminator:
                # unfreeze_model_parameters(model.discriminator)
                # freeze_model_parameters(model.text_m)
                # freeze_model_parameters(model.speech_m)
                for _ in range(0, args.d_steps):
                    batch = batch_getter.get_discriminator_batch()
                    train_discriminator_step(losses, model, batch, args.d_steps)
                optimizer_step(model, optimizer, args)


            if args.model_type == "transformer":
                adjust_learning_rate(optimizer, args.e_in, total_step + 1)
        if args.model_type == "rnn":
            sched.step()

        # Eval and save
        per, eval_losses = evaluate(model, valid_dataloader)
        log_loss_metrics(losses, epoch)
        log_loss_metrics(eval_losses, epoch, eval=True)
        model.teacher.step()
        print("Eval_ epoch {:-3d} PER {:0.3f}\%".format(epoch, per*100))
        save_ckp(epoch, per, model, optimizer, per < best, args.checkpoint_path)
        if per < best:
            print("\t Best score - saving model!")
            best = per
    model.eval()
    return model

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

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_transformer, drop_last=True, num_workers=16)
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
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_transformer, drop_last=True, num_workers=16)
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
        discriminator = Discriminator(args.hidden)
    model = UNAST(text_m, speech_m, discriminator, teacher).to(DEVICE)

    # initialize optimizer
    optimizer = None
    if args.optim_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    s_epoch, best = 0, 100

    if args.load_path is not None:
        s_epoch, best, model, optimizer = load_ckp(args.load_path, model, optimizer)
        s_epoch = s_epoch + 1

    model.teacher.iter = s_epoch
    return s_epoch, best, model, optimizer

def initialize_datasets(args):
    # TODO: replace these if we want
    print("#### Getting Dataset ####")
    supervised_train_dataset = get_dataset('labeled_train.csv')
    unsupervised_train_dataset = get_dataset('unlabeled_train.csv')
    val_dataset = get_dataset('val.csv')
    full_train_dataset = get_dataset('full_train.csv')
    return supervised_train_dataset, unsupervised_train_dataset, val_dataset, full_train_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='JSON config files')
    args = parse_with_config(parser)

    global DEVICE
    DEVICE = init_device(args)

    train(args)
