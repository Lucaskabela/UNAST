'''
Contains the code for training the encoder/decoders, including:
    - Cross model loss
    - Supervised loss
    - Denoising loss
    - Discriminator loss
'''
from utils import set_seed, parse_with_config
from preprocess import get_dataset, DataLoader, collate_fn_transformer
from module import TextPrenet, TextPostnet, RNNDecoder, RNNEncoder
from network import AutoEncoderNet
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

def evaluate():
    raise Exception("Not implemented yet!")

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
    
    # init models and optimizers
    text_pre = TextPrenet(args.phoneme_embedding_size, args.hidden_size)
    encoder = RNNEncoder(args.hidden_size, args.hidden_size, args.latent_size, bidirectional=False)
    decoder = RNNDecoder(args.latent_size, args.hidden_size, args.hidden_size, args.decoder_out, attention=False)
    text_post = TextPostnet(args.decoder_out, args.hidden_size)

    model = AutoEncoderNet(text_pre, encoder, decoder, text_post)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,  weight_decay=1e-5)

    global_step = 0

    for epoch in range(args.epochs):
        losses = []
        for data in dataloader:
            character, mel, mel_input, pos_text, pos_mel, text_len = data
            encoder_outputs, latent_hidden = model.encode(character)
            # TODO: compute context masks for sequence here, something like `c_mask = character.ne(0)`
            # TODO: Add <SOS>, <EOS>, and <PAD> special chars (already in symbols?) and add <SOS>, <EOS> to examples
            # TODO: Fix the model api!!!
            output = model.decode(latent_hidden, encoder_outputs)
            loss = F.cross_entropy(output, character)
            
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(loss.detach().cpu().numpy())
            if train_log is not None:
                train_log.add_scalar("loss", losses[-1], global_step)
            global_step += 1
        # with torch.no_grad():
            # evaluate(model, valid_dataset)
        # TODO: Add evaluation code
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
