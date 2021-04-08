'''
Contains the code for the encoder & decoders, which are composed of modules.  
These modules are pre, encoder, a decoder, a post, and optional discriminator.
    - BiRNN encoder/decoder for text and speech (with attention (?))
    - Transformer encoder/decoder for text and speech
'''

import torch.nn as nn
from module import *

class AutoEncoderNet(nn.Module):
    # Speech net will use speech prenet/postnet, similarly with text
    def __init__(prenet, encoder, decoder, postnet):
        super(AutoEncoderNet, self).__init__()
        self.prenet = prenet
        self.encoder = encoder
        self.decoder = decoder
        self.postnet = postnet
    
    # TODO: Need more input for padded sequences?
    def preprocess(self, raw_input_tensor):
        '''
        Preprocesses raw input using the prenet, returning input for the encoder

        Args:
            - raw_input_tensor: a tensor of shape [batch_size x seq_len x dim]
                + For text, dim will be 1 and tensor type will be Long, 
                  corresponding to the indices for phoneme embeddings
                + For speech, dim will be the ???
        
        Returns:
            - input_tensor: a tensor of shape [batch_size x seq_len x hidden_dim]
                where hidden dim is the expected input shape of the encoder
        '''
        return self.prenet(raw_input_tensor)

    def encode(self, raw_input_tensor):
        '''
        Encodes raw input using the prenet & encoder, returning latent vector & 
        necessary components for attention

        Args:
            - raw_input_tensor: a tensor of shape [batch_size x seq_len x dim]
                + For text, dim will be 1 and tensor type will be Long, 
                  corresponding to the indices for phoneme embeddings
                + For speech, dim will be the ???
        
        Returns:
            - latent_tensor: a tensor of shape [batch_size x 1 x latent_dim], which
                is the encoded representation of the input sequence
            - attention_stuff: any and every args for attention, not sure what this
                looks like yet
        '''
        return self.encoder(self.preprocess(raw_input_tensor))

    def decode(self, embed_decode, hidden_state, enc_output, enc_ctxt_mask):
        '''
        Decodes a latent_tensor using the decoder & postnet, returning output for 
        evaluation

        Args:
            - latent_tensor: a tensor of shape [batch_size x 1 x latent_dim]
            - attention_stuff: any variables needed for computing attention
            - teacher_sequence: a tensor of shape [batch_size x teacher_len x 1]
                which is of type Long.  Contains info needed for seq2seq decoding
        
        Returns:
            - output_tensor: a tensor of shape [batch_size x teacher_len x dim]
                + For text, dim will be #phonemes which is a distribution over 
                    phonemes
                + For speech, dim will be the ??? which is something log filter related
        '''
        # TODO: Change this API ever so slightly
        return self.postprocess(self.decoder(embed_decode, hidden_state, enc_output, enc_ctxt_mask))

    def postprocess(self, decoded_latent):
        '''
        Applies some post processing onto the network, which returns appropriate output

        Args:
            - decoded_latent: a tensor of shape [batch_size x teacher_len x out_dim]
                which is the raw output of the network
        Returns:
            - output_tensor: a tensor of shape [batch_size x teacher_len x res_dim]
                + For text, res_dim will be #phonemes which is a distribution over 
                    phonemes
                + For speech, res_dim will be the ??? which is something log filter related 
        '''
        return self.postnet(decoded_latent)



class SpeechNet(AutoEncoderNet):
    # For the purpose of good OO design, make the nets parameters for now.
    # Can change this later
    def __init__(self, speech_prenet, speech_encoder, speech_decoder, speech_postnet):
        super(SpeechNet, self).__init__(speech_prenet, speech_encoder, speech_decoder, speech_postnet)
    
    # TODO: Decide what methods go here if any


class TextNet(AutoEncoderNet):
    # For the purpose of good OO design, make the nets parameters for now.
    # Can change this later
    def __init__(self, text_prenet, text_encoder, text_decoder, text_postnet):
        super(TextNet, self).__init__(text_prenet, text_encoder, text_decoder, text_postnet)
    
    # TODO: Decide what methods go here if any

class Discriminator(nn.Module):
    # TODO: Fill in with linear layers :) 
    def __init__(self, enc_dim, hidden, out_classes=2, dropout=.2):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(enc_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_classes)
        self.dropout = nn.Dropout(p=dropout)
        self.non_linear = nn.LeakyRelu(.2)

    def forward(self, enc_output):
        temp = self.dropout(self.non_linear(self.fc1(enc_output)))
        temp2 = self.dropout(self.non_linear(self.fc2(temp)))
        return self.fc3(temp2)
    
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
