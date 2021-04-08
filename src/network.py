'''
Contains the code for the encoder & decoders, which are composed of modules.  
These modules are pre, encoder, a decoder, a post, and optional discriminator.
    - BiRNN encoder/decoder for text and speech (with attention (?))
    - Transformer encoder/decoder for text and speech
'''

import torch.nn as nn
from module import *
from utils import PAD_IDX, SOS_IDX,  EOS_IDX
import random

class AutoEncoderNet(nn.Module):
    # Speech net will use speech prenet/postnet, similarly with text
    def __init__(self):
        super(AutoEncoderNet, self).__init__()
    
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
            - pad_mask: a tensor of shape [batch_size x seq_len] with 1s for padded 
                indices and 0 for everything else
        '''
        raise Exception("Please use a subclass for text or speech")

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
            Depends on subclass
        '''
        raise Exception("Please use a subclass for text or speech")

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
        raise Exception("Please use a subclass for text or speech")

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
        raise Exception("Please use a subclass for text or speech")



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
    
class SpeechTransformer(AutoEncoderNet):
    # TODO: Fill in with pre/post needed and enc/dec
    def __init__(self, args):
        super(SpeechTransformer, self).__init__()


class SpeechRNN(AutoEncoderNet):
    # TODO: Fill in with pre/post needed and enc/dec
    def __init__(self, args):
        super(SpeechRNN, self).__init__()


class TextTransformer(AutoEncoderNet):
    # TODO: Fill in with pre/post needed and enc/dec
    def __init__(self, args):
        super(TextTransformer, self).__init__()


class TextRNN(AutoEncoderNet):
    # TODO: Fill in with pre/post needed and enc/dec
    def __init__(self, args):
        super(TextRNN, self).__init__()
        self.prenet = TextPrenet(args.embed_dim, args.e_in)
        self.encoder = RNNEncoder(args.e_in, args.hidden, args.e_out, 
            dropout=args.e_p, num_layers=args.e_layers, bidirectional=bool(args.e_bi))
        self.decoder = RNNDecoder(args.e_out, args.d_in, args.hidden, args.d_out, 
            dropout=args.d_p, num_layers=args.d_layers, attention=bool(args.d_attn))
        self.postnet = TextPostnet(args.d_out, args.hidden)

    def preprocess(self, raw_input_tensor):
        # raw_input_tensor should be a LongTensor of size [batch_size x seq_len x 1]
        # should already be padded as well
        # Get a mask of 0 for no padding, 1 for padding of size [batch_size x seq_len]
        pad_mask = raw_input_tensor.eq(PAD_IDX).float().squeeze() 
        return self.prenet(raw_input_tensor), pad_mask

    def encode(self, raw_input_tensor):
        """
        Returns:
            - output: a tensor of shape [batch_size x seq_len x latent_dim], which
                is the projected hidden state from each timestep in the sequence
            - h_t: the (h_n, c_n) pair from the last timestep of the network.  
                Dimension varies depending on num_layers, but should
                be proportional to  [num_layers, batch, hidden_size]
            - pad_mask: a tensor of shape [batch_size x seq_len] with 1s for padded 
                indices and 0 for everything else
        """
        embedded_phonemes, pad_mask = self.preprocess(raw_input_tensor)
        enc_output, hidden_state = self.encoder(embedded_phonemes)
        return enc_output, hidden_state, pad_mask

    def decode_sequence(self, target, hidden_state, enc_output, enc_ctxt_mask, teacher_ratio=1):
        """
        For easier training!  target is teacher forcing [batch_size x seq_len]
        """
        batch_size, max_out_len = target.shape[0], target.shape[1]
        outputs = []
        input_ = torch.from_numpy(np.asarray([SOS_IDX for i in range(0, batch_size)]))
        for i in range(max_out_len):
            input_embed = self.prenet.embed(input_).unsqueeze(1)
            dec_out, hidden_state = self.decode(input_embed, hidden_state, enc_output, enc_ctxt_mask)
            outputs.append(dec_out)
            if random.random() < teacher_ratio:
                input_ = target[:, i]
            else:
                input_ = torch.argmax(dec_out, dim=-1).squeeze()
        return torch.stack(outputs, dim=1).squeeze(2)

    def decode(self, embed_decode, hidden_state, enc_output, enc_ctxt_mask):
        """
        NOTE: embed_deode should be [batch_size x 1 x d_in] as this is autoregressive 
            decoding (seq2seq).
        Returns:
            - dec_output: a tensor of shape [batch_size x teacher_len x dim]
                + For text, dim will be #phonemes which is a distribution over 
                    phonemes (needs to be log_softmaxed!)
                + For speech, dim will be the ??? which is something log filter related
            - dec_hidden: (h_n, c_n) from decoder LSTM
        """
        dec_output, dec_hidden = self.decoder(embed_decode, hidden_state, enc_output, enc_ctxt_mask)
        return self.postprocess(dec_output), dec_hidden

    def postprocess(self, dec_output, distrib=False):
        
        final_out = self.postnet(dec_output)
        if distrib:
            # If we want to return distribution, take log softmax!
            return F.log_softmax(final_out, dim=-1)
        else:
            return final_out
