'''
Contains the code for the encoder & decoders, which are composed of modules.
These modules are pre, encoder, a decoder, a post, and optional discriminator.
    - BiRNN encoder/decoder for text and speech (with attention (?))
    - Transformer encoder/decoder for text and speech
'''

import torch.nn as nn
from module import *
from utils import PAD_IDX, SOS_IDX,  EOS_IDX, sent_lens_to_mask, noise_fn
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
                + For speech, dim will be the number of mel filters, likely 80

        Returns:
            - input_tensor: a tensor of shape [batch_size x seq_len x hidden_dim]
                where hidden dim is the expected input shape of the encoder
            - pad_mask: a tensor of shape [batch_size x seq_len] with 0s for padded
                indices and 1s for everything else
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
                + For speech, dim will be the number of mel filters, likely 80

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


class UNAST(nn.Module):
    def __init__(self, text_m, speech_m, discriminator=None, teacher=None):
        """NOTE: text_m and speech_m should be same type (RNN or Transformer)"""
        super(UNAST, self).__init__()
        self.text_m = text_m
        self.speech_m = speech_m
        self.discriminator = discriminator
        self.teacher = teacher

    def text_ae(self, character_input, ret_enc_hid=False):
        return self.text_m.forward(character_input, noise_in=True, teacher_ratio=self.teacher.get_val(), ret_enc_hid=ret_enc_hid)

    def speech_ae(self, mel, ret_enc_hid=False):
        return self.speech_m.forward(mel, noise_in=True, ret_enc_hid=ret_enc_hid, teacher_ratio=self.teacher.get_val())

    def cm_text_in(self, character_input, ret_enc_hid=False):
        t_e_o, t_hid, t_pad_mask = self.text_m.encode(character_input)
        pred, _ = self.speech_m.infer_sequence(t_hid, t_e_o, t_pad_mask)
        cm_s_e_o, cm_s_hid, cm_s_pad_mask = self.speech_m.encode(pred)
        text_pred = self.text_m.decode_sequence(character_input, cm_s_hid, cm_s_e_o, 
            cm_s_pad_mask, teacher_ratio=self.teacher.get_val())
        if ret_enc_hid:
            return text_pred, t_hid, cm_s_hid
        return text_pred

    def cm_speech_in(self, mel, ret_enc_hid=False):
        s_e_o, s_hid, s_pad_mask = self.speech_m.encode(mel)
        text_pred = self.text_m.infer_sequence(s_hid, s_e_o, s_pad_mask)
        cm_t_e_o, cm_t_hid, cm_t_pad_mask = self.text_m.encode(text_pred)
        pred, stop_pred = self.speech_m.decode_sequence(mel, cm_t_hid, cm_t_e_o, 
            cm_t_pad_mask, teacher_ratio=self.teacher.get_val())
        if ret_enc_hid:
            return pred, stop_pred, s_hid, cm_t_hid
        return pred, stop_pred

    def tts(self, character, mel, infer=False, ret_enc_hid=False):
        t_e_o, t_hid, t_pad_mask = self.text_m.encode(character)
        if not infer:
            pred, stop_pred = self.speech_m.decode_sequence(mel, t_hid, t_e_o, 
                t_pad_mask, teacher_ratio=self.teacher.get_val())
        else:
            pred, stop_pred = self.speech_m.infer_sequence(t_hid, t_e_o, t_pad_mask)
        if ret_enc_hid:
            return pred, stop_pred, t_hid
        return pred, stop_pred

    def asr(self, character, mel, infer=False, ret_enc_hid=False):
        s_e_o, s_hid, s_pad_mask = self.speech_m.encode(mel)
        if not infer:
            text_pred = self.text_m.decode_sequence(character, s_hid, s_e_o, 
                s_pad_mask, teacher_ratio=self.teacher.get_val())
        else:
            text_pred = self.text_m.infer_sequence(s_hid, s_e_o, s_pad_mask)
        if ret_enc_hid:
            return text_pred, s_hid
        return text_pred

    def num_params(self):
        pytorch_total_params = 0
        for p in self.parameters():
            if p.requires_grad:
                pytorch_total_params += p.numel()
        return pytorch_total_params

class Discriminator(nn.Module):
    # From Lample et al.
    # "3 hidden layers, 1024 hidden layers, smoothing coefficient 0.1"
    def __init__(self, enc_dim, hidden=1024, out_classes=2, dropout=.2, relu=.2):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(enc_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, out_classes)
        self.dropout = nn.Dropout(p=dropout)
        self.non_linear = nn.LeakyReLU(relu)

    def forward(self, enc_output):
        temp = self.dropout(self.non_linear(self.fc1(enc_output)))
        temp2 = self.dropout(self.non_linear(self.fc2(temp)))
        temp3 = self.dropout(self.non_linear(self.fc3(temp2)))
        return self.fc4(temp3)

class SpeechTransformer(AutoEncoderNet):
    # TODO: Fill in with pre/post needed and enc/dec
    def __init__(self, args):
        super(SpeechTransformer, self).__init__()


class SpeechRNN(AutoEncoderNet):
    # TODO: Fill in with pre/post needed and enc/dec
    def __init__(self, args):
        super(SpeechRNN, self).__init__()
        self.prenet = SpeechPrenet(args.num_mels, args.s_pre_hid, args.e_in, p=args.s_pre_drop)
        self.encoder = RNNEncoder(args.e_in, args.hidden, dropout=args.e_drop,
            num_layers=args.num_layers, bidirectional=args.e_bi)
        self.decoder = RNNDecoder(args.e_in, args.hidden * self.encoder.num_dir, args.hidden, dropout=args.d_drop, 
            num_layers=args.num_layers, attention=args.d_attn, attn_dim=args.attn_dim)
        self.postnet = SpeechPostnet(args.num_mels, args.hidden, p=args.s_post_drop)


    def preprocess(self, raw_input_tensor):
        # raw_input_tensor should be a LongTensor of size [batch_size x seq_len x 1]
        # should already be padded as well
        # Get a mask of 0 for no padding, 1 for padding of size [batch_size x seq_len]
        # NOTE: I do not think this is correct for mel... but check if all are padding in num_mels
        pad_mask = torch.all(raw_input_tensor.eq(PAD_IDX), dim=-1)
        return self.prenet(raw_input_tensor), pad_mask

    def encode(self, raw_input_tensor, noise_in=False):
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
        mel_features, pad_mask = self.preprocess(raw_input_tensor)
        if noise_in:
            mel_features = noise_fn(mel_features)
        enc_output, hidden_state = self.encoder(mel_features)
        return enc_output, hidden_state, pad_mask

    def infer_sequence(self, hidden_state, enc_output, enc_ctxt_mask, max_len=815):

        batch_size = enc_output.shape[0]
        outputs = []
        stops = []
        stop_lens = torch.zeros(batch_size, device=enc_output.device)

        # get a all 0 frame for first timestep
        input_ = torch.zeros((batch_size, 1, self.postnet.num_mels), device=enc_output.device)
        i = 0
        keep_gen = torch.any(stop_lens.eq(0)) and i < max_len
        if self.decoder.las:
            self.decoder.attention_layer.init_memory(enc_output)

        while keep_gen:
            (dec_out, stop_pred), hidden_state = self.decode(input_, hidden_state, enc_output, enc_ctxt_mask)
            stop_pred = stop_pred.squeeze()
            stops.append(stop_pred)
            # set stop_lens here!
            outputs.append(dec_out)
            input_ = outputs[-1]
            i += 1

            # double check this!
            stop_mask = (torch.sigmoid(stop_pred) >= .5).logical_and(stop_lens == 0)
            stop_lens[stop_mask] = i
            keep_gen = torch.any(stop_lens.eq(0)) and i < max_len

        if self.decoder.las:
            self.decoder.attention_layer.clear_memory()

        # Maybe this is a bit overkil...
        stop_lens[stop_lens == 0] = len(outputs)
        pad_mask = sent_lens_to_mask(stop_lens, len(outputs))

        res, res_stop = torch.stack(outputs, dim=1).squeeze(2), torch.stack(stops, dim=1).squeeze(1)
        res = (res + self.postprocess(res)) * pad_mask.unsqueeze(-1).detach()
        res_stop = res_stop * pad_mask.detach()
        return res, res_stop

    def decode_sequence(self, target, hidden_state, enc_output, enc_ctxt_mask, teacher_ratio=1):
        """
        For easier training!  target is teacher forcing [batch_size x seq_len x num_mels]
        """
        batch_size, max_out_len = target.shape[0], target.shape[1]
        outputs = []
        stops = []
        # get a all 0 frame for first timestep
        input_ = torch.zeros((batch_size, 1, self.postnet.num_mels), device=enc_output.device)
        if self.decoder.las:
            self.decoder.attention_layer.init_memory(enc_output)
        for i in range(max_out_len):
            (dec_out, stop_pred), hidden_state = self.decode(input_, hidden_state, enc_output, enc_ctxt_mask)
            outputs.append(dec_out)
            stops.append(stop_pred)

            if random.random() < teacher_ratio:
                input_ = target[:, i, :].unsqueeze(1)
            else:
                input_ = outputs[-1]

        if self.decoder.las:
            self.decoder.attention_layer.clear_memory()

        decoder_outputs = torch.stack(outputs, dim=1).squeeze()
        res = decoder_outputs + self.postprocess(decoder_outputs)
        return res, torch.stack(stops, dim=1).squeeze()

    def decode(self, input_, hidden_state, enc_output, enc_ctxt_mask):
        """
        NOTE: input_ should be [batch_size x 1 x num_mels] as this is autoregressive
            decoding (seq2seq).
        Returns:
            - dec_output: a tensor of shape [batch_size x 1 x num_mels]
                + For text, dim will be #phonemes which is a distribution over
                    phonemes (needs to be log_softmaxed!)
                + For speech, dim will be the ??? which is something log filter related
            - dec_hidden: (h_n, c_n) from decoder LSTM
        """
        input_ = self.prenet(input_)
        dec_output, dec_hidden = self.decoder(input_, hidden_state, enc_output, enc_ctxt_mask)
        return self.postnet.mel_and_stop(dec_output), dec_hidden

    def postprocess(self, dec_output, distrib=False):
        return self.postnet(dec_output)

    def forward(self, mel, noise_in=False, ret_enc_hid=False, teacher_ratio=1):
        encoder_outputs, latent_hidden, pad_mask = self.encode(mel, noise_in=noise_in)
        pred, stop_pred = self.decode_sequence(mel, latent_hidden, encoder_outputs, pad_mask, teacher_ratio)
        if ret_enc_hid:
            return pred, stop_pred, latent_hidden
        return pred, stop_pred

class TextTransformer(AutoEncoderNet):
    # TODO: Fill in with pre/post needed and enc/dec
    def __init__(self, args):
        super(TextTransformer, self).__init__()


class TextRNN(AutoEncoderNet):
    # TODO: Fill in with pre/post needed and enc/dec
    def __init__(self, args):
        super(TextRNN, self).__init__()
        self.prenet = TextPrenet(args.t_emb_dim, args.e_in, p=args.t_pre_drop)
        self.encoder = RNNEncoder(args.e_in, args.hidden, dropout=args.e_drop,
            num_layers=args.num_layers, bidirectional=args.e_bi)
        self.decoder = RNNDecoder(args.t_emb_dim, args.hidden * self.encoder.num_dir, args.hidden, dropout=args.d_drop, 
            num_layers=args.num_layers, attention=args.d_attn, attn_dim=args.attn_dim)
        self.postnet = TextPostnet(args.hidden, args.t_post_drop)

    def preprocess(self, raw_input_tensor):
        # raw_input_tensor should be a LongTensor of size [batch_size x seq_len x 1]
        # should already be padded as well
        # Get a mask of 0 for no padding, 1 for padding of size [batch_size x seq_len]
        pad_mask = raw_input_tensor.eq(PAD_IDX) 
        return self.prenet(raw_input_tensor), pad_mask

    def encode(self, raw_input_tensor, noise_in=False):
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
        if noise_in:
            embedded_phonemes = noise_fn(embedded_phonemes)
        enc_output, hidden_state = self.encoder(embedded_phonemes)
        return enc_output, hidden_state, pad_mask

    def decode_sequence(self, target, hidden_state, enc_output, enc_ctxt_mask, teacher_ratio=1):
        """
        For easier training!  target is teacher forcing [batch_size x seq_len]
        """
        batch_size, max_out_len = target.shape[0], target.shape[1]
        outputs = []
        input_ = torch.tensor([SOS_IDX for i in range(0, batch_size)], device=enc_output.device, dtype=torch.long)
        if self.decoder.las:
            self.decoder.attention_layer.init_memory(enc_output)
        for i in range(max_out_len):
            dec_out, hidden_state = self.decode(input_, hidden_state, enc_output, enc_ctxt_mask)
            outputs.append(dec_out)
            if random.random() < teacher_ratio:
                input_ = target[:, i]
            else:
                input_ = torch.argmax(dec_out, dim=-1).squeeze()
        if self.decoder.las:
            self.decoder.attention_layer.clear_memory()
        return torch.stack(outputs, dim=1).squeeze(2)

    def decode(self, input_, hidden_state, enc_output, enc_ctxt_mask):
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
        input_embed = self.prenet.emb_dropout(self.prenet.embed(input_).unsqueeze(1))
        dec_output, dec_hidden = self.decoder(input_embed, hidden_state, enc_output, enc_ctxt_mask)
        return self.postprocess(dec_output), dec_hidden

    def postprocess(self, dec_output, distrib=False):

        final_out = self.postnet(dec_output)
        if distrib:
            # If we want to return distribution, take log softmax!
            return F.log_softmax(final_out, dim=-1)
        else:
            return final_out

    def infer_sequence(self, hidden_state, enc_output, enc_ctxt_mask, max_len=300):

        batch_size = enc_output.shape[0]
        outputs = []
        seq_lens = torch.zeros(batch_size, device=enc_output.device)

        # get a all SOS for first timestep
        input_ = torch.tensor([SOS_IDX for i in range(0, batch_size)], device=enc_output.device, dtype=torch.long)
        i = 0
        keep_gen = torch.any(seq_lens.eq(0)) and i < max_len
        if self.decoder.las:
            self.decoder.attention_layer.init_memory(enc_output)
        while keep_gen:
            dec_out, hidden_state = self.decode(input_, hidden_state, enc_output, enc_ctxt_mask)
            input_ = torch.argmax(dec_out, dim=-1)
            # set stop_lens here!
            outputs.append(input_)
            input_ = input_.squeeze()
            i += 1

            # double check this!
            stop_mask = (input_ == EOS_IDX).logical_and(seq_lens == 0)
            seq_lens[stop_mask] = i
            keep_gen = torch.any(seq_lens.eq(0)) and i < max_len

        # Maybe this is a bit overkil...
        if self.decoder.las:
            self.decoder.attention_layer.clear_memory()
        seq_lens[seq_lens == 0] = len(outputs)
        pad_mask = sent_lens_to_mask(seq_lens, len(outputs)).detach()

        res = torch.stack(outputs, dim=1).squeeze(2)
        res = res * pad_mask
        return res

    def forward(self, input_, noise_in=False, teacher_ratio=1, ret_enc_hid = False):
        encoder_outputs, latent_hidden, pad_mask = self.encode(input_, noise_in=noise_in)
        pred = self.decode_sequence(input_, latent_hidden, encoder_outputs, pad_mask, teacher_ratio)
        if ret_enc_hid:
            return pred, latent_hidden
        return pred
