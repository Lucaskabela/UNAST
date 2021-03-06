'''
Contains the nn.Modules which compose the networks. This includes modules
for preprocessing speech and text, the transformer & RNN encoder/decoder, & post
processing modules for text and speech.
'''
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import math
from utils import PAD_IDX

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
    the hidden size of Transformer.
    """
    def __init__(self, num_mels, hidden_size, output_size, p=0.5):
        """
        :param num_mels: number of mel filters
        :param hidden_size: dimension of hidden unit
        :param output_size: dimension of output
        :param p: dropout probability (zero-out probability)
        """
        super(SpeechPrenet, self).__init__()
        self.input_size = num_mels
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer = nn.Sequential(OrderedDict([
             ('fc1', Linear(self.input_size, self.hidden_size)),
             ('relu1', nn.ReLU()),
             ('dropout2', nn.Dropout(p)),
             ('fc2', Linear(self.hidden_size, self.output_size)),
             ('relu2', nn.ReLU()),
             ('dropout2', nn.Dropout(p)),
        ]))

    def forward(self, input_):
        """
        :param input_: Tensor of mel-spectrogram input (input_mel),
                       these should be of dimensions [batch, length, num_mels]
        """
        out = self.layer(input_)
        return out


class SpeechPostnet(nn.Module):
    """
    PostNet for Speech Decoder copied from Transformer-TTS.

    There are 5 layers of 1D convolutions as specified in Ren's
    paper, and this convolution network aims to refine the
    quality of the generated mel-spectrograms.
    """
    def __init__(self, num_mels, num_hidden, p=0.1):
        """
        :param num_mels: number of mel filters
        :param num_hidden: dimension of hidden unit
        :param p: dropout probability (zero-out probability)
        """
        super(SpeechPostnet, self).__init__()
        self.num_mels = num_mels
        self.conv1 = Conv(in_channels=num_mels,
                          out_channels=num_hidden,
                          kernel_size=5,
                          padding=4,
                          w_init='tanh')
        self.conv_list = nn.ModuleList([Conv(in_channels=num_hidden,
                                             out_channels=num_hidden,
                                             kernel_size=5,
                                             padding=4,
                                             w_init='tanh')
                                        for _ in range(3)])
        self.conv2 = Conv(in_channels=num_hidden,
                          out_channels=num_mels,
                          kernel_size=5,
                          padding=4)

        self.batch_norm_list = nn.ModuleList([nn.BatchNorm1d(num_hidden)
                                                  for _ in range(3)])
        self.pre_batchnorm = nn.BatchNorm1d(num_hidden)

        self.dropout1 = nn.Dropout(p=p)
        self.dropout_list = nn.ModuleList([nn.Dropout(p=p) for _ in range(3)])

        self.stop_linear = nn.Linear(num_hidden, 1)
        self.linear_project = nn.Linear(num_hidden, num_mels)

    def forward(self, input_):
        """
        :param input_: Tensor of mel-spectrogram output from decoder,
                       these should be of dimensions [batch, length, num_mels]
                       as we change the shape ourselves
        """
        # Causal Convolution (for auto-regressive)
        input_ = input_.permute(0, 2, 1)
        input_ = self.dropout1(torch.tanh(self.pre_batchnorm(self.conv1(input_)[:, :, :-4])))
        for batch_norm, conv, dropout in zip(self.batch_norm_list, self.conv_list, self.dropout_list):
            input_ = dropout(torch.tanh(batch_norm(conv(input_)[:, :, :-4])))
        input_ = self.conv2(input_)[:, :, :-4]
        input_ = input_.permute(0, 2, 1)
        return input_

    def mel_and_stop(self, decoder_out):
        return self.linear_project(decoder_out), self.stop_linear(decoder_out)


class TextPrenet(nn.Module):
    """
    Prenet for Text Encoder copied from Transformer-TTS.

    Essentially, maps phoneme IDs to an embedding space using
    convolutional networks. Should be the same as Ren's paper
    since it outputs an embedding of size 256 for each phoneme.
    """
    def __init__(self, embedding_size, num_hidden, p=0.5):
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

        self.emb_dropout = nn.Dropout(p=p)
        self.dropout1 = nn.Dropout(p=p)
        self.dropout2 = nn.Dropout(p=p)
        self.dropout3 = nn.Dropout(p=p)

    def forward(self, input_):
        """
        :param input_: Tensor of Phoneme IDs (text)
        """
        input_ = self.emb_dropout(self.embed(input_))
        return self.forward_fcn(input_)
    
    def forward_fcn(self, input_):
        input_ = input_.transpose(1, 2)
        input_ = self.dropout1(torch.relu(self.batch_norm1(self.conv1(input_))))
        input_ = self.dropout2(torch.relu(self.batch_norm2(self.conv2(input_))))
        input_ = self.dropout3(torch.relu(self.batch_norm3(self.conv3(input_))))
        input_ = input_.transpose(1, 2)
        return input_


class TextPostnet(nn.Module):
    """

    """
    def __init__(self, hidden, p=.2):
        super(TextPostnet, self).__init__()
        self.fc1 = nn.Linear(hidden, len(symbols))
        self.dropout1 = nn.Dropout(p=p)

    def forward(self, decode_out):
        """
        Need to do log softmax if you want probabilities
        """
        return self.fc1(self.dropout1(decode_out))

# Taken from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        # d_model is model dimension
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model_scale = math.sqrt(d_model)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * self.d_model_scale + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self, ninp, nhead, nhid, dropout, nlayers):
        super(TransformerEncoder, self).__init__()
        encoder_layers = torch.nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers)

    def forward(self, src, src_mask, src_pad_mask):
        # src will be (N, S, E) -> needs to be (S, N, E) for transformer!
        src = src.transpose(0, 1)
        memory_out = self.transformer_encoder(src, src_mask, src_pad_mask)
        return memory_out.transpose(0, 1)


class TransformerDecoder(nn.Module):
    def __init__(self, ninp, nhead, ffn_dim, dropout, nlayers):
        super(TransformerDecoder, self).__init__()
        decoder_layer = torch.nn.TransformerDecoderLayer(ninp, nhead, ffn_dim, dropout)
        self.transformer_decoder = torch.nn.TransformerDecoder(decoder_layer, nlayers)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt = tgt.transpose(0, 1)
        memory = memory.transpose(0, 1)
        out = self.transformer_decoder(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
        return out.transpose(0, 1)



class RNNEncoder(nn.Module):
    # TODO: Write this
    def __init__(self, d_in, hidden, dropout=.2, num_layers=1, bidirectional=False):
        super(RNNEncoder, self).__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        self.num_dir = 2 if bidirectional else 1

        # TODO: expirement with something else than LSTM
        self.rnn = nn.LSTM(d_in, hidden, num_layers=num_layers,
            bidirectional=bidirectional, batch_first=True, dropout=dropout)

        if self.num_dir == 2:
            self.reduce_h_W = nn.Linear(hidden * 2, hidden, bias=True)
            self.reduce_c_W = nn.Linear(hidden * 2, hidden, bias=True)

    def forward(self, sequence, lengths):

        packed_seq = pack_padded_sequence(sequence, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, hn = self.rnn(packed_seq)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, padding_value=PAD_IDX)
        if self.num_dir == 2:
            # Potential source of error here!!
            h = hn[0].view(self.num_layers, self.num_dir, -1, self.hidden)
            c = hn[1].view(self.num_layers, self.num_dir, -1, self.hidden)

            # Cat the representations from forward and backward LSTMs
            # NOTE: This indexing is robust to num_layers & bidirectional.  Pls no change
            h_ = torch.cat((h[:, 0, :, :], h[:, 1, :, :]), dim=-1)
            c_ = torch.cat((c[:, 0, :, :], c[:, 1, :, :]), dim=-1)

            # Now reduce!
            new_h = self.reduce_h_W(h_)
            new_c = self.reduce_c_W(c_)
            h_t = (new_h, new_c)
        else:
            h, c = hn[0][:], hn[1][:]
            h_t = (h, c)

        return output, h_t



class RNNDecoder(nn.Module):
    def __init__(self, d_in, enc_out_size, hidden, dropout=.2, num_layers=1, attention=None, attn_dim=0):
        super(RNNDecoder, self).__init__()

        self.attention = attention
        if self.attention:
            self.input_size = enc_out_size + d_in
        else:
            self.input_size = d_in

        self.rnn = nn.LSTM(self.input_size, hidden, num_layers=num_layers,
            batch_first=True, dropout=dropout)

        if self.attention is not None:
            if self.attention == "lsa":
                self.attention_layer = LocationSensitiveAttention(hidden, enc_out_size, attn_dim)
            elif self.attention == "luong":
                self.attention_layer = LuongGeneralAttention(hidden, enc_out_size, attn_dim)
            self.linear_projection = Linear(enc_out_size + hidden, hidden, w_init='tanh')
            self.dropout1 = nn.Dropout(p=dropout)


    def forward(self, embed_decode, hidden_state, enc_output, enc_ctxt_mask):
        if self.attention is not None:
            # Handles num_layers > 1 by taking last layer
            hidden_key = hidden_state[0][-1].unsqueeze(0)
            attn_W = self.attention_layer(hidden_key, enc_output, enc_ctxt_mask)
            decode_input = torch.cat((embed_decode, attn_W), dim=-1)
        else:
            decode_input = embed_decode

        output, hidden = self.rnn(decode_input, hidden_state)
        if self.attention:
            output = self.dropout1(torch.tanh(self.linear_projection(torch.cat((output, attn_W), dim=-1))))
        return output, hidden


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = Conv(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = Linear(attention_n_filters, attention_dim,
                                         bias=False, w_init='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention

class LocationSensitiveAttention(nn.Module):
    def __init__(self, hidden_dim, encoder_dim, attention_dim,
                 attention_location_n_filters=32, attention_location_kernel_size=31):
        super(LocationSensitiveAttention, self).__init__()
        self.query_layer = Linear(hidden_dim, attention_dim,
                                      bias=False, w_init='tanh')
        self.memory_layer = Linear(encoder_dim, attention_dim, bias=False,
                                       w_init='tanh')
        self.v = Linear(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def init_memory(self, enc_output):
        self.processed_memory = self.memory_layer(enc_output)
        self.attention_weights_cum = torch.zeros((enc_output.shape[0], enc_output.shape[1]),
            device=enc_output.device)
        self.attention_weights = torch.zeros((enc_output.shape[0], enc_output.shape[1]),
            device=enc_output.device)

    def clear_memory(self):
        self.processed_memory = None
        self.attention_weights_cum = None
        self.attention_weight = None

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (1x batch x n_mel_channels)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)
        RETURNS
        -------
        alignment (batch, max_time)
        """
        processed_query = self.query_layer(query).permute(1, 0, 2)
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, hidden_state, memory, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs (done once for compute)
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data - 1 for no padding, 0 for padding
        """
        attention_cat = torch.cat((self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)

        alignment = self.get_alignment_energies(
            hidden_state, self.processed_memory, attention_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        self.attention_weights = F.softmax(alignment, dim=1)
        self.attention_weights_cum += self.attention_weights
        ctxt = torch.bmm(self.attention_weights.unsqueeze(1), memory)
        return ctxt



class LuongGeneralAttention(nn.Module):
    def __init__(self, hidden_size, enc_out_size, attention_dim):
        super(LuongGeneralAttention, self).__init__()
        self.hidden_size = hidden_size
        self.enc_out_size = enc_out_size
        self.project_hid = nn.Linear(hidden_size, attention_dim, bias=False)
        self.project_eo = nn.Linear(enc_out_size, attention_dim, bias=False)
        self.fc2 = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, hidden, enc_output, enc_ctxt_mask):
        '''
        Returns the alignment weights
        '''
        src_len = enc_output.shape[1]
        e_o = enc_output.permute(1, 0, 2)
        combined = self.project_hid(hidden) + self.project_eo(e_o)

        # combined is [seq_len x batch_size x hidden + enc_out]
        align_scores = self.fc2(torch.tanh(combined)).squeeze(-1)
        # align_scores is [seq_len x batch_size], so flip and mask all padding
        align_scores = align_scores.permute(1, 0)
        align_scores.data.masked_fill_(enc_ctxt_mask, -np.inf)

        align_weights = F.softmax(align_scores, dim=-1).unsqueeze(1)
        # align_weights is [batch_size x seq_len x 1] where each entry is score over sequence

        # Note: If input is a (b??n??m) tensor, mat2 is a (b??m??p) tensor
        #  --- out will be a (b??n??p) tensor.
        # And we want a [batch_size x 1 x enc_out_size], so we have the right order
        ctxt = torch.bmm(align_weights, enc_output)
        return ctxt


class Highwaynet(nn.Module):
    """
    Highway network, copied from Transformer-TTS.

    This is used by the CBHG network.
    """
    def __init__(self, num_units, num_layers=4):
        """
        :param num_units: dimension of hidden unit
        :param num_layers: # of highway layers
        """
        super(Highwaynet, self).__init__()
        self.num_units = num_units
        self.num_layers = num_layers
        self.gates = nn.ModuleList()
        self.linears = nn.ModuleList()
        for _ in range(self.num_layers):
            self.linears.append(Linear(num_units, num_units))
            self.gates.append(Linear(num_units, num_units))

    def forward(self, input_):
        out = input_

        # highway gated function
        for fc1, fc2 in zip(self.linears, self.gates):
            h = torch.relu(fc1.forward(out))
            t_ = torch.sigmoid(fc2.forward(out))
            c = 1. - t_
            out = h * t_ + out * c

        return out


class CBHG(nn.Module):
    """
    CBHG Module, copied from Transformer-TTS.

    This is used in the post-processing model for speech synthesis.
    The mel-spectrograms are passed through this to convert into
    magnitude spectrograms, which are then converted back to wavs.
    """
    def __init__(self, hidden_size, K=16, projection_size=256, num_gru_layers=2, max_pool_kernel_size=2):
        """
        :param hidden_size: dimension of hidden unit
        :param K: # of convolution banks
        :param projection_size: dimension of projection unit
        :param num_gru_layers: # of layers of GRUcell
        :param max_pool_kernel_size: max pooling kernel size
        """
        super(CBHG, self).__init__()
        self.hidden_size = hidden_size
        self.projection_size = projection_size
        self.convbank_list = nn.ModuleList()
        self.convbank_list.append(nn.Conv1d(in_channels=projection_size,
                                                out_channels=hidden_size,
                                                kernel_size=1,
                                                padding=int(np.floor(1/2))))

        for i in range(2, K+1):
            self.convbank_list.append(nn.Conv1d(in_channels=hidden_size,
                                                out_channels=hidden_size,
                                                kernel_size=i,
                                                padding=int(np.floor(i/2))))

        self.batchnorm_list = nn.ModuleList()
        for i in range(1, K+1):
            self.batchnorm_list.append(nn.BatchNorm1d(hidden_size))

        convbank_outdim = hidden_size * K

        self.conv_projection_1 = nn.Conv1d(in_channels=convbank_outdim,
                                             out_channels=hidden_size,
                                             kernel_size=3,
                                             padding=int(np.floor(3 / 2)))
        self.conv_projection_2 = nn.Conv1d(in_channels=hidden_size,
                                               out_channels=projection_size,
                                               kernel_size=3,
                                               padding=int(np.floor(3 / 2)))
        self.batchnorm_proj_1 = nn.BatchNorm1d(hidden_size)

        self.batchnorm_proj_2 = nn.BatchNorm1d(projection_size)


        self.max_pool = nn.MaxPool1d(max_pool_kernel_size, stride=1, padding=1)
        self.highway = Highwaynet(self.projection_size)
        self.gru = nn.GRU(self.projection_size, self.hidden_size // 2, num_layers=num_gru_layers,
                          batch_first=True,
                          bidirectional=True)


    def _conv_fit_dim(self, x, kernel_size=3):
        if kernel_size % 2 == 0:
            return x[:,:,:-1]
        else:
            return x

    def forward(self, input_):
        input_ = input_.contiguous()
        batch_size = input_.size(0)
        total_length = input_.size(-1)

        convbank_list = list()
        convbank_input = input_

        # Convolution bank filters
        for k, (conv, batchnorm) in enumerate(zip(self.convbank_list, self.batchnorm_list)):
            convbank_input = torch.relu(batchnorm(self._conv_fit_dim(conv(convbank_input), k+1).contiguous()))
            convbank_list.append(convbank_input)

        # Concatenate all features
        conv_cat = torch.cat(convbank_list, dim=1)

        # Max pooling
        conv_cat = self.max_pool(conv_cat)[:,:,:-1]

        # Projection
        conv_projection = torch.relu(self.batchnorm_proj_1(self._conv_fit_dim(self.conv_projection_1(conv_cat))))
        conv_projection = self.batchnorm_proj_2(self._conv_fit_dim(self.conv_projection_2(conv_projection))) + input_

        # Highway networks
        highway = self.highway.forward(conv_projection.transpose(1,2))

        # Bidirectional GRU
        self.gru.flatten_parameters()
        out, _ = self.gru(highway)

        return out
