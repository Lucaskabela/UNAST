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
import torch.nn.functional as F

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

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='JSON config files')
    args = parse_with_config(parser)
    train(args)
