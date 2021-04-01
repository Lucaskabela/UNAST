'''
Contains the code for the encoder & decoders, which are composed of modules.  
These modules are pre, encoder, a decoder, a post, and optional discriminator.
    - BiRNN encoder/decoder for text and speech (with attention (?))
    - Transformer encoder/decoder for text and speech
'''

import torch.nn as nn

class SpeechNet(nn.Module):
    # For the purpose of good OO design, make the nets parameters for now.
    # Can change this later
    def __init__(self, speech_prenet, speech_encoder, speech_decoder, speech_postnet):
        super(SpeechNet, self).__init__()
    
    # TODO: Decide what methods go here.  Probably an encode/decode


class TextNet(nn.Module):
    # For the purpose of good OO design, make the nets parameters for now.
    # Can change this later
    def __init__(self, text_prenet, text_encoder, text_decoder, text_postnet):
        super(TextNet, self).__init__()
    
    # TODO: Decide what methods go here.  Probably an encode/decode

class Discriminator(nn.Module):
    # TODO: Fill in with linear layers :) 
    def __init__(self):
        super(Discriminator, self).__init__()
        
class SpeechTransformer(SpeechNet):
    # TODO: Fill in with pre/post needed and enc/dec
    def __init_(self):
        super(SpeechTransformer, self).__init__()


class SpeechRNN(SpeechNet):
    # TODO: Fill in with pre/post needed and enc/dec
    def __init_(self):
        super(SpeechRNN, self).__init__()


class TextTransformer(TextNet):
    # TODO: Fill in with pre/post needed and enc/dec
    def __init_(self):
        super(TextTransformer, self).__init__()


class TextRNN(TextNet):
    # TODO: Fill in with pre/post needed and enc/dec
    def __init_(self):
        super(TextRNN, self).__init__()
