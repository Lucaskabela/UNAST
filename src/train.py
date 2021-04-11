'''
Contains the code for training the encoder/decoders, including:
    - Cross model loss
    - Supervised loss
    - Denoising loss
    - Discriminator loss
'''
from utils import set_seed, parse_with_config, PAD_IDX, init_device
from preprocess import get_dataset, DataLoader, collate_fn_transformer
from module import TextPrenet, TextPostnet, RNNDecoder, RNNEncoder
from network import TextRNN, SpeechRNN
from tqdm import tqdm
import audio_parameters as ap
import argparse
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

DEVICE = init_device()

# TODO: Refactor these losses...
def loss_fn(output, target, speech=False):
    '''
        Computes the NLL loss between output and target
    '''
    PAD_IDX = 0
    if speech:
        return F.mse_loss(output, target)
    else:
        return F.cross_entropy(output, target, ignore_index=PAD_IDX)


def discriminator_loss(output, target):
    return F.cross_entropy(output, target)


def eval_text_auto(target, hidden_state, enc_output, enc_ctxt_mask, text_model):
    res = text_model.infer_sequence(hidden_state, enc_output, enc_ctxt_mask, max_len=target.shape[1])
    loss = loss_fn(res, target).item()
    return loss

def eval_speech_auto(target, hidden_state, enc_output, enc_ctxt_mask, text_model):
    res = text_model.infer_sequence(hidden_state, enc_output, enc_ctxt_mask, max_len=target.shape[1])
    loss = loss_fn(res, target).item()
    return loss

def evaluate(text_model, speech_model, valid_dataset):
    """
        Expect validation set to have paired speech & text!
        We evaluate on 4 metrics:  autoencoder text loss, autoencoder speech loss,
            ASR (evaluated by PER), and TTS (evaluated by MSE). 
    """
    text_model.eval()
    speech_model.eval()
    with torch.no_grad():
        ae_text_loss, ae_speech_loss, per, tts_err = 0, 0, 0, 0
        avg_constant = 0
        for data in valid_dataset:
            character, mel, mel_input, pos_text, pos_mel, text_len = data
            character = character.to(DEVICE)
            mel = mel.to(DEVICE)
            enc_text, text_hidden_state, text_pad_mask = text_model.encode(character)
            enc_speech, speech_hidden_state, speech_pad_mask = speech_model.encode(mel)

            # TODO: decoding here!
            decoded_text = None

            ae_text_loss += eval_text_auto(character, text_hidden_state, enc_text, enc_ctxt_mask, text_model)
            ae_speech_loss += autoencoder_loss(decoded_speech, mel_input).detach().item()
            #per += crossmodel_loss().item()
            #tts_err += crossmodel_loss().item()
            avg_constant += 1

    text_model.train()
    speech_model.train()
    ae_text_loss, ae_speech_loss = ae_text_loss / avg_constant, ae_speech_loss / avg_constant
    per, tts_err = per / avg_constant, tts_err / avg_constant
    return ae_text_loss, ae_speech_loss, per, tts_err

def train_text_auto(args):
    '''
    TODO:
        1. DONE -- Get Dataset
        2. Init models & optimizers
        3. Train, or for each epoch:
                For each batch:
                - Choose which loss function
                - Freeze appropriate networks
                - Run through networks
                - Get loss
                - Update
                Metrics (like validation, etc)
        4. Return trained text & speech for validation, metric measurement

        TODO: Include functionality for saving, loading from save
    '''
    set_seed(args.seed)

    # TODO: Replace get_dataset() with getting train/valid/test split
    dataset = get_dataset()
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
            pred = model.forward(character).permute(1, 2, 0)
            
            char_ = character.permute(1, 0)
            loss = F.cross_entropy(pred_, char_, ignore_index=PAD_IDX)
            
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(loss.detach().cpu().item())
            if train_log is not None:
                train_log.add_scalar("loss", losses[-1], global_step)
            global_step += 1

        # evaluate(model, valid_dataset)

        avg_l = np.mean(losses)
        print("epoch %-3d \t loss = %0.3f \t" % (epoch, avg_l))
        # if validation < best:
        # TODO: Add model (and optimizer) saving for reloading training
        #     print("Saving model!")
        #     best = validation
        #     model.save_model()
    return model

def train_speech_auto(args):
    '''
    TODO:
        1. Get Dataset
        2. Init models & optimizers
        3. Train, or for each epoch:
                For each batch:
                - Choose which loss function
                - Freeze appropriate networks
                - Run through networks
                - Get loss
                - Update
                Metrics (like validation, etc)
        4. Return trained text & speech for validation, metric measurement

        TODO: Include functionality for saving, loading from save
    '''
    set_seed(args.seed)

    # TODO: Replace get_dataset() with getting train/valid/test split
    dataset = get_dataset()
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

            
            loss = F.mse_loss(pred, mel)
            
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(loss.detach().cpu().item())
            if train_log is not None:
                train_log.add_scalar("loss", losses[-1], global_step)
            global_step += 1

        # evaluate(model, valid_dataset)

        avg_l = np.mean(losses)
        print("epoch %-3d \t loss = %0.3f \t" % (epoch, avg_l))
        # if validation < best:
        # TODO: Add model (and optimizer) saving for reloading training
        #     print("Saving model!")
        #     best = validation
        #     model.save_model()
    return model

def train(args):
    set_seed(args.seed)
    print("#### Getting Dataset ####")
    supervised_train_dataset = get_dataset('labeled_train.csv')
    unsupervised_train_dataset = get_dataset('unlabeled_train.csv')
    val_dataset = get_dataset('val.csv')
    full_train_dataset = get_dataset('full_train.csv')
    # init models and optimizers
    model = None
    optimizer = None

    for epoch in range(args.epochs):
        # Training model
        supervised_dataloader = DataLoader(supervised_train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_transformer, drop_last=True, num_workers=16)
        supervised_iter = iter(supervised_dataloader)

        unsupervised_dataloader = DataLoader(unsupervised_train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_transformer, drop_last=True, num_workers=16)

        bar = tqdm(unsupervised_dataloader)
        bar.set_description("Epoch {} Training Model".format(epoch))

        # step counter for a single epoch
        # used to determine which training task to do
        epoch_step = 0

        # We are considering one pass through the unsupervised dataset as
        # one epoch
        for unsupervised_batch in bar:
            epoch_step += 1
            if epoch_step % 3 == 0:
                # Janky way to check to do a supervised step
                try:
                    supervised_batch = supervised_iter.next()
                except StopIteration:
                    supervised_iter = iter(DataLoader(supervised_train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_transformer, drop_last=True, num_workers=16))
                    supervised_batch = supervised_iter.next()
                # TODO: Do a supervised step with supervised_batch

                # This enforces that we still use the unsupervised_batch in this
                # iteration
                epoch_step += 1
            if epoch_step % 3 == 1:
                # TODO: Do a denoising step with this unsupervised_batch
                pass
            if epoch_step % 3 == 2:
                # TODO: Do a unsupervised cross modal step
                # with this unsupervised_batch
                pass

        if args.train_discriminator:
            # Train discriminator
            discriminator_dataloader = DataLoader(full_train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_transformer, drop_last=True, num_workers=16)

            bar = tqdm(discriminator_dataloader)
            bar.set_description("Epoch {} Training Discriminator".format(epoch))
            for batch in bar:
                # TODO: Freeze model
                # TODO: Train discriminator over the whole train dataset
                pass

        avg_l = np.mean(losses)
        print("epoch %-3d \t loss = %0.3f \t" % (epoch, avg_l))
        # if validation < best:
        # TODO: Add model (and optimizer) saving for reloading training
        #     print("Saving model!")
        #     best = validation
        #     model.save_model()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='JSON config files')
    # TODO: clean up the parser/initialization of models
    args = parse_with_config(parser)
    train_text_auto(args)
