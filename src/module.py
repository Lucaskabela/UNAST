'''
Contains the nn.Modules which compose the networks. This includes modules 
for preprocessing speech and text, the transformer & RNN encoder/decoder, & post
processing modules for text and speech.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

from data.symbols import symbols

class Linear(nn.Module):
    """
    Linear Module copied from Transformer-TTS.
    
    A linear layer initialized using the Xavier Uniform method.
    """
    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        """
        :param in_dim: dimension of input
        :param out_dim: dimension of output
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        return self.linear_layer(x)

class Conv(nn.Module):
    """
    Convolution Module copied from Transformer-TTS.
    
    A 1D convolution layer initialized using the Xavier Uniform
    method.
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, bias=True, w_init='linear'):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              bias=bias)

        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        x = self.conv(x)
        return x

class SpeechPrenet(nn.Module):
    """
    Prenet for Speech Encoder copied from Transformer-TTS.
    
    As described in Ren's paper, 2-layer dense-connected network
    with hidden size of 256, and the output dimension equals to
    the hidden size of Transformer. Note that the dropout is
    removed here, which differs from what's in Transformer-TTS.
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        :param input_size: dimension of input
        :param hidden_size: dimension of hidden unit
        :param output_size: dimension of output
        """
        super(SpeechPrenet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer = nn.Sequential(OrderedDict([
             ('fc1', Linear(self.input_size, self.hidden_size)),
             ('relu1', nn.ReLU()),
             ('fc2', Linear(self.hidden_size, self.output_size)),
             ('relu2', nn.ReLU()),
        ]))

    def forward(self, input_):
        out = self.layer(input_)
        return out

class SpeechPostnet(nn.Module):
    # TODO: Fill in from TTS repo :) 
    def __init__(self):
        super(SpeechPostnet, self).__init__()

class TextPrenet(nn.Module):
    """
    Prenet for Text Encoder copied from Transformer-TTS.
    
    Essentially, maps phoneme IDs to an embedding space using
    convolutional networks. Should be the same as Ren's paper
    since it outputs an embedding of size 256 for each phoneme.
    """
    def __init__(self, embedding_size, num_hidden):
        """
        :param embedding_size: phoneme embedding size
        :param num_hidden: output embedding size
        """
        super(TextPrenet, self).__init__()
        self.embedding_size = embedding_size
        self.embed = nn.Embedding(len(symbols), embedding_size, padding_idx=0)

        self.conv1 = Conv(in_channels=embedding_size,
                          out_channels=num_hidden,
                          kernel_size=5,
                          padding=int(np.floor(5 / 2)),
                          w_init='relu')
        self.conv2 = Conv(in_channels=num_hidden,
                          out_channels=num_hidden,
                          kernel_size=5,
                          padding=int(np.floor(5 / 2)),
                          w_init='relu')

        self.conv3 = Conv(in_channels=num_hidden,
                          out_channels=num_hidden,
                          kernel_size=5,
                          padding=int(np.floor(5 / 2)),
                          w_init='relu')

        self.batch_norm1 = nn.BatchNorm1d(num_hidden)
        self.batch_norm2 = nn.BatchNorm1d(num_hidden)
        self.batch_norm3 = nn.BatchNorm1d(num_hidden)

        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        self.projection = Linear(num_hidden, num_hidden)

    def forward(self, input_):
        input_ = self.embed(input_) 
        input_ = input_.transpose(1, 2) 
        input_ = self.dropout1(torch.relu(self.batch_norm1(self.conv1(input_)))) 
        input_ = self.dropout2(torch.relu(self.batch_norm2(self.conv2(input_)))) 
        input_ = self.dropout3(torch.relu(self.batch_norm3(self.conv3(input_)))) 
        input_ = input_.transpose(1, 2) 
        input_ = self.projection(input_) 
        return input_

class TextPostnet(nn.Module):
    # TODO: Fill in from TTS repo :) 
    def __init__(self):
        super(TextPostnet, self).__init__()

# TODO: Consider adding Encoder/Decoder base class here

class TransformerEncoder(nn.Module):
    # TODO: Fill in from TTS repo :) 
    def __init__(self):
        super(TransformerEncoder, self).__init__()

class TransformerDecoder(nn.Module):
    # TODO: Fill in from TTS repo :) 
    def __init__(self):
        super(TransformerDecoder, self).__init__()

class RNNEncoder(nn.Module):
    # TODO: Write this
    def __init__(self):
        super(RNNEncoder, self).__init__()

class RNNDecoder(nn.Module):
    # TODO: Write this
    def __init__(self):
        super(RNNDecoder, self).__init__()

