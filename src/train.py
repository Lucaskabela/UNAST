'''
Contains the code for training the encoder/decoders, including:
    - Cross model loss
    - Supervised loss
    - Denoising loss
    - Discriminator loss
'''
from utils import set_seed, parse_with_config, PAD_IDX
from preprocess import get_dataset, DataLoader, collate_fn_transformer
from module import TextPrenet, TextPostnet, RNNDecoder, RNNEncoder
from network import TextRNN
from tqdm import tqdm
import audio_parameters as ap
import argparse
import torch.nn.functional as F
import torch.nn as nn
import torch


# TODO: Refactor these losses...
def autoencoder_loss(output, target, speech=False):
    '''
        Computes the NLL loss between output and target
    '''
    PAD_IDX = 0
    if speech:
        return F.mse_loss(output, target)
    else:
        return F.cross_entropy(output, target, ignore_index=PAD_IDX)

def supervised_loss(output, target):
    PAD_IDX = 0
    if speech:
        return F.mse_loss(output, target)
    else:
        return F.cross_entropy(output, target, ignore_index=PAD_IDX)

def crossmodel_loss(output, target):
    PAD_IDX = 0
    if speech:
        return F.mse_loss(output, target)
    else:
        return F.cross_entropy(output, target, ignore_index=PAD_IDX)

def discriminator_loss(output, target):
    return F.cross_entropy(output, target)

def evaluate(text_model, speech_model, valid_dataset):
    """
        Expect validation set to have paired speech & text!
        We evaluate on 4 metrics:  autoencoder text loss, autoencdoer speech loss,
            ASR (evaluated by PER), and TTS (evaluated by MSE). 
    """
    text_model.eval()
    speech_model.eval()
    with torch.no_grad():
        ae_text_loss, ae_speech_loss, per, tts_err = 0, 0, 0, 0
        avg_constant = 0
        for data in valid_dataset:
            character, mel, mel_input, pos_text, pos_mel, text_len = data

            enc_text, text_hidden_state, text_pad_mask = text_model.encode(character)

            # TODO: decoding here!
            decoded_text = None

            ae_text_loss += autoencoder_loss(decoded_text, character).item()
            ae_speech_loss += autoencoder_loss(decoded_speech, mel_input).detach().item()
            #per += crossmodel_loss().item()
            #tts_err += crossmodel_loss().item()
            avg_constant += 1

    text_model.train()
    speech_model.train()
    ae_text_loss, ae_speech_loss = ae_text_loss / avg_constant, ae_speech_loss / avg_constant
    per, tts_err = per / avg_constant, tts_err / avg_constant
    return ae_text_loss, ae_speech_loss, per, tts_err

def train(args):
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
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_transformer, drop_last=True, num_workers=16)

    model = TextRNN(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,  weight_decay=1e-5)

    global_step = 0

    for epoch in range(args.epochs):
        losses = []
        for data in dataloader:
            character, mel, mel_input, pos_text, pos_mel, text_len = data
            encoder_outputs, latent_hidden, pad_mask = model.encode(character)
            print(text_len.shape)
            pred = model.decode_sequence(character, latent_hidden, encoder_outputs, pad_mask)
            
            loss = F.cross_entropy(pred, character, ignore_index=PAD_IDX)
            
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='JSON config files')
    # TODO: clean up the parser/initialization of models
    args = parse_with_config(parser)
    train(args)
