'''
Contains the code for training the encoder/decoders, including:
    - Cross model loss
    - Supervised loss
    - Denoising loss
    - Discriminator loss
'''
from utils import set_seed, parse_with_config
from preprocess import get_dataset, DataLoader, collate_fn_transformer
from tqdm import tqdm
import audio_parameters as ap
import argparse

def autoencoder_loss():
    raise Exception("Not implemented yet!")

def supervised_loss():
    raise Exception("Not implemented yet!")

def crossmodel_loss():
    raise Exception("Not implemented yet!")

def discriminator_loss():
    raise Exception("Not implemented yet!")

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
    dataset = get_dataset()
    # init models and optimizers
    model = None
    optimizer = None

    for epoch in range(args.epochs):
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_transformer, drop_last=True, num_workers=16)

        pbar = tqdm(dataloader)
        for i, data in enumerate(pbar):
            pbar.set_description("Processing at epoch %d"%epoch)

            character, mel, mel_input, pos_text, pos_mel, _ = data

        for batch in dataloader:
            # choose loss function here!
            model.decode(model.encode(batch))
        evaluate(model, valid_dataset)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='JSON config files')
    args = parse_with_config(parser)
    train(args)
