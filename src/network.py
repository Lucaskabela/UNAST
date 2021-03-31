'''
Contains the code for the encoder & decoders, which are composed of modules.  
These modules are pre, encoder, a decoder, a post, and optional discriminator.
    - BiRNN encoder/decoder for text and speech (with attention (?))
    - Transformer encoder/decoder for text and speech
'''

import torch.nn as nn

class Discriminator(nn.Module):
    # TODO: Fill in with linear layers :) 
    def __init__(self):
        super(Discriminator, self).__init__()
        
class SpeechTransformer(nn.Module):
    # TODO: Fill in with pre/post needed and enc/dec
    def __init_(self):
        super(SpeechTransformer, self).__init__()
    
class TextTransformer(nn.Module):
    # TODO: Fill in with pre/post needed and enc/dec
    def __init_(self):
        super(TextTransformer, self).__init__()

class TextRNN(nn.Module):
    # TODO: Fill in with pre/post needed and enc/dec
    def __init_(self):
        super(TextRNN, self).__init__()

class SpeechRNN(nn.Module):
    # TODO: Fill in with pre/post needed and enc/dec
    def __init_(self):
        super(SpeechRNN, self).__init__()
