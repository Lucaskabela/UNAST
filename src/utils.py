'''
Contains any and all code we didnt want to put somewhere else
'''
import argparse
def parse_args():
    parser = argparse.ArgumentParser(from_file_prefix_chars="@")

    parser.add_argument('--seed', type=int, default=0, help='RNG seed (default = 0)')
    parser.add_argument('--epochs', type=int, default=100, help='num epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    
    return parser.parse_args('@hyperparams.txt')