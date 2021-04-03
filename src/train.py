'''
Contains the code for training the encoder/decoders, including:
    - Cross model loss
    - Supervised loss
    - Denoising loss
    - Discriminator loss
'''
from utils import parse_args, set_seed

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

def train():
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
    args = parse_args()
    set_seed(args.seed)
    # Get dataset
    train_dataset, valid_dataset = None

    # init models and optimizers
    model = None
    optimizer = None

    for epoch in range(args.epochs):
        for batch in dataset:
            # choose loss function here!
            model.decode(model.encode(batch))
        evaluate(model, valid_dataset)

    return model