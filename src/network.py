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
    def preprocess(self, input_, input_lens):
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

    def encode(self, input_, input_lens, noise_in=False):
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

    def text_ae(self, text, text_len):
        return self.text_m.forward(text, text_len, noise_in=True, teacher_ratio=self.teacher.get_val())
    
    def speech_ae(self, mel, mel_len):
        return self.speech_m.forward(mel, mel_len, noise_in=True, teacher_ratio=self.teacher.get_val())

    def cm_text_in(self, text, text_len):
        t_e_o, t_mask = self.text_m.encode(text, text_len)
        pred, _, pred_lens = self.speech_m.infer_sequence(t_e_o, t_mask)
        cm_s_e_o, cm_mask = self.speech_m.encode(pred, pred_lens)
        text_pred = self.text_m.decode_sequence(text, text_len, cm_s_e_o, cm_mask,
            teacher_ratio=self.teacher.get_val())
        return text_pred

    def cm_speech_in(self, mel, mel_len):
        s_e_o, s_mask = self.speech_m.encode(mel, mel_len)
        text_pred, text_pred_len = self.text_m.infer_sequence(s_e_o, s_mask)
        cm_t_e_o, cm_t_masks = self.text_m.encode(text_pred, text_pred_len)
        pred, stop_pred = self.speech_m.decode_sequence(mel, mel_len, cm_t_e_o, 
            cm_t_masks, teacher_ratio=self.teacher.get_val())
        return pred, stop_pred
    
    def tts(self, text, text_len, mel, mel_len, infer=False):
        t_e_o, t_masks = self.text_m.encode(text, text_len)
        if not infer:
            pred, stop_pred = self.speech_m.decode_sequence(mel, mel_len, t_e_o, 
                t_masks, teacher_ratio=self.teacher.get_val())
        else:
            pred, stop_pred, _ = self.speech_m.infer_sequence(t_e_o, t_masks)
        return pred, stop_pred

    def asr(self, text, text_len, mel, mel_len, infer=False):
        s_e_o, s_masks = self.speech_m.encode(mel, mel_len)
        if not infer:
            return self.text_m.decode_sequence(text, text_len, s_e_o, 
                s_masks, teacher_ratio=self.teacher.get_val())
        else:
            return self.text_m.infer_sequence(s_e_o, s_masks)

    def num_params(self):
        pytorch_total_params = 0
        for p in self.parameters():
            if p.requires_grad:
                pytorch_total_params += p.numel()
        return pytorch_total_params

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

    def __init__(self, args):
        super(SpeechTransformer, self).__init__()
        self.prenet = SpeechPrenet(args.num_mels, args.s_pre_hid, args.e_in, p=args.s_pre_drop)
        self.pos_emb = PositionalEncoding( args.e_in)
        self.encoder = TransformerEncoder( args.e_in, args.nhead, args.hidden, args.e_drop, args.num_layers)
        self.decoder = TransformerDecoder( args.e_in, args.nhead, args.hidden, args.d_drop, args.num_layers)
        self.postnet = SpeechPostnet(args.num_mels, args.hidden, p=args.s_post_drop)

    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_mask(x, x_lens, enc=True):
        max_seq_len = x.shape[1]
        if enc:
            x_mask = torch.zeros((max_seq_len, max_seq_len), device=DEVICE, dtype=torch.bool)
        else:
            x_mask = generate_square_subsequent_mask(max_seq_len)

        pad_mask = torch.logical_not(sent_lens_to_mask(x_lens, max_seq_len))
        print(pad_mask)
        return x_mask, pad_mask

    def preprocess(self, input_, input_lens, enc=True):
        input_mask, input_pad_mask = create_mask(input_, input_lens, enc=True)
        pre_in =  self.prenet(input_)
        return self.pos_emb(pre_in) (input_mask, input_pad_mask)

    def encode(self, input_, input_lens, noise_in=False):
        if noise_in:
            input_ = noise_fn(input_)
        embedded_input, (input_mask, input_pad_mask) = self.preprocess(input_, input_lens)
        enc_outputs = self.encoder(embedded_input, input_mask, input_pad_mask)
        return enc_outputs, (input_mask, input_pad_mask)

    def decode(self, tgt, tgt_lens, enc_outputs, enc_mask):
        embedded_tgt, (tgt_mask, tgt_pad_mask) = self.preprocess(tgt, tgt_lens, enc=False)
        return self.postnet.mel_and_stop(self.decoder(embedded_tgt, enc_outputs, tgt_mask, None, tgt_pad_mask, enc_mask))

    def postprocess(self, decoded_latent):
        return self.postnet(dec_output)

    def infer_sequence(self, memory, masks, max_len=815):
        input_mask, input_pad_mask = masks
        batch_size = enc_output.shape[0]
        outputs = torch.zeros((batch_size, 1, self.postnet.num_mels), device=memory.device)
        stop_lens = torch.full((batch_size,), max_len, device=memory.device)
        stops = torch.zeros((batch_size, 1), device=memory.device)
        i = 0
        keep_gen = torch.any(stop_lens.eq(max_len)) and i < max_len
        while keep_gen:
            (dec_out, stop_pred) = self.decode(outputs, stop_lens, memory, input_pad_mask)
            stops = torch.cat([stops, stop_pred.squeeze(2)], dim=1)
            outputs = torch.cat([outputs, dec_out], dim=1)

            # set stop_lens here!
            stop_mask = (torch.sigmoid(stop_pred.squeeze()) >= .5).logical_and(stop_lens == max_len)
            stop_lens[stop_mask] = i
            i +=1
            keep_gen = torch.any(stop_lens.eq(max_len)) and i < max_len

        # Maybe this is a bit overkil...
        res, res_stop = (outputs + self.postprocess(outputs))[:, 1:, :], stops[:, 1:]
        pad_mask = sent_lens_to_mask(stop_lens, res.shape[1])
        return res * pad_mask.unsqueeze(-1), res_stop * pad_mask, stop_lens

    def decode_sequence(self, tgt, tgt_lens, enc_outputs, masks, teacher_ratio=1):
        # No use for teacher ratio here...

        input_mask, input_pad_mask = masks

        # Need to adjust tgt with "go" padding:
        sos = torch.zeros((batch_size, 1, self.postnet.num_mels), device=enc_outputs.device)
        tgt_input = torch.cat([sos, tgt[:, :-1, :]], dim=1)

        embedded_tgt, (tgt_mask, tgt_pad_mask) = self.preprocess(tgt_input, tgt_lens, enc=False)
        outs = self.decoder(embedded_tgt, enc_outputs, tgt_mask, None,
                                        tgt_pad_mask, input_pad_mask)

        (dec_out, stop_pred)  = self.postnet.mel_and_stop(outs)
        return dec_out + self.postprocess(dec_out), stop_pred.squeeze(2)

    def forward(self, mel, mel_len, noise_in=False, teacher_ratio=1):
        enc_outputs, masks = self.encode(mel, mel_len, noise_in)
        return self.decode_sequence(mel, mel_len, enc_outputs, masks)
        

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
    

    def preprocess(self, mel, mel_lens):
        max_seq_len = mel.shape[1]
        pad_mask = torch.logical_not(sent_lens_to_mask(mel_lens, max_seq_len))
        return self.prenet(mel), pad_mask

    def encode(self, mel, mel_lens, noise_in=False):
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
        if noise_in:
            mel = noise_fn(mel)
        mel_features, pad_mask = self.preprocess(mel, mel_lens)
        enc_output, hidden_state = self.encoder(mel_features)
        return (hidden_state, enc_output), pad_mask

    def infer_sequence(self, enc_outputs, enc_ctxt_mask, max_len=815):

        hidden_state, enc_output = enc_outputs

        batch_size = enc_output.shape[0]
        outputs = torch.zeros((batch_size, 1, self.postnet.num_mels), device=enc_output.device)
        stops = torch.zeros((batch_size, 1), device=enc_output.device)
        stop_lens = torch.full((batch_size,), max_len, device=enc_output.device)
        
        # get a all 0 frame for first timestep
        i = 0
        keep_gen = torch.any(stop_lens.eq(max_len)) and i < max_len
        if self.decoder.attention == "lsa":
            self.decoder.attention_layer.init_memory(enc_output)
        while keep_gen:
            (dec_out, stop_pred), hidden_state = self.decode(outputs[:, -1, :].unsqueeze(1), hidden_state, enc_output, enc_ctxt_mask)
            stops = torch.cat([stops, stop_pred.squeeze(2)], dim=1)
            outputs = torch.cat([outputs, dec_out], dim=1)

            # double check this!
            stop_mask = (torch.sigmoid(stop_pred.squeeze()) >= .5).logical_and(stop_lens == max_len)
            stop_lens[stop_mask] = i
            i += 1
            keep_gen = torch.any(stop_lens.eq(max_len)) and i < max_len
        if self.decoder.attention == "lsa":
            self.decoder.attention_layer.clear_memory()

        # Maybe this is a bit overkil...
        res, res_stop = (outputs + self.postprocess(outputs))[:, 1:, :], stops[:, 1:]
        pad_mask = sent_lens_to_mask(stop_lens, res.shape[1])
        return res * pad_mask.unsqueeze(-1), res_stop * pad_mask, stop_lens


    def decode_sequence(self, target, target_len, enc_outputs, enc_ctxt_mask, teacher_ratio=1):
        """
        For easier training!  target is teacher forcing [batch_size x seq_len x num_mels]
        """
        hidden_state, enc_output = enc_outputs
        batch_size, max_out_len = target.shape[0], target.shape[1]
        outputs = torch.zeros((batch_size, 1, self.postnet.num_mels), device=enc_output.device)
        input_ = outputs[:, -1, :].unsqueeze(1)
        stops = torch.zeros((batch_size, 1), device=enc_output.device)
        # get a all 0 frame for first timestep
        if self.decoder.attention == "lsa":
            self.decoder.attention_layer.init_memory(enc_output)
        for i in range(max_out_len):
            (dec_out, stop_pred), hidden_state = self.decode(input_, hidden_state, enc_output, enc_ctxt_mask)
            stops = torch.cat([stops, stop_pred.squeeze(2)], dim=1)
            outputs = torch.cat([outputs, dec_out], dim=1)

            if random.random() < teacher_ratio:
                input_ = target[:, i, :].unsqueeze(1)
            else:
                input_ = outputs[:, -1, :].unsqueeze(1)
        if self.decoder.attention == "lsa":
            self.decoder.attention_layer.clear_memory()

        res = outputs + self.postprocess(outputs)
        return res[:, 1:, :], stops[:, 1:]

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

    def forward(self, mel, mel_len, noise_in=False, teacher_ratio=1):
        encoder_outputs, pad_mask = self.encode(mel, mel_len, noise_in=noise_in)
        pred, stop_pred = self.decode_sequence(mel, mel_len, encoder_outputs, pad_mask, teacher_ratio)
        return pred, stop_pred
        
class TextTransformer(AutoEncoderNet):
    # TODO: Fill in with pre/post needed and enc/dec
    def __init__(self, args):
        super(TextTransformer, self).__init__()
        self.prenet = TextPrenet(args.t_emb_dim, args.e_in, p=args.t_pre_drop)
        self.pos_emb = PositionalEncoding( args.e_in)
        self.encoder = TransformerEncoder( args.e_in, args.nhead, args.hidden, args.e_drop, args.num_layers)
        self.decoder = TransformerDecoder( args.e_in, args.nhead, args.hidden, args.d_drop, args.num_layers)
        self.postnet = TextPostnet(args.hidden, args.t_post_drop)

    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_mask(x, x_lens, enc=True):
        max_seq_len = x.shape[1]
        if enc:
            x_mask = torch.zeros((max_seq_len, max_seq_len), device=DEVICE, dtype=torch.bool)
        else:
            x_mask = generate_square_subsequent_mask(max_seq_len)

        pad_mask = torch.logical_not(sent_lens_to_mask(x_lens, max_seq_len))
        return x_mask, pad_mask

    def preprocess(self, input_, input_lens, enc=True):
        input_mask, input_pad_mask = create_mask(input_, input_lens, enc=True)
        pre_in =  self.prenet(input_)
        return self.pos_emb(pre_in) (input_mask, input_pad_mask)

    def encode(self, input_, input_lens, noise_in=False):
        if noise_in:
            input_ = noise_fn(input_)
        embedded_input, (input_mask, input_pad_mask) = self.preprocess(input_, input_lens)
        enc_outputs = self.encoder(embedded_input, input_mask, input_pad_mask)
        return enc_outputs, (input_mask, input_pad_mask)

    def decode(self, tgt, tgt_lens, enc_outputs, enc_mask):
        embedded_tgt, (tgt_mask, tgt_pad_mask) = self.preprocess(tgt, tgt_lens, enc=False)
        return self.postprocess(self.decoder(embedded_tgt, enc_outputs, tgt_mask, None, tgt_pad_mask, enc_mask))

    def postprocess(self, decoded_latent):
        return self.postnet(dec_output)

    def infer_sequence(self, memory, masks, max_len=815):
        input_mask, input_pad_mask = masks
        batch_size = memory.shape[0]
        outputs = torch.as_tensor([SOS_IDX for i in range(0, batch_size)], device=memory.device, dtype=torch.long).unsqueeze(1)
        stop_lens = torch.full((batch_size,), max_len,  device=memory.device)

        i = 0
        keep_gen = torch.any(stop_lens.eq(max_len)) and i < max_len
        while keep_gen:
            dec_out = self.decode(outputs, stop_lens, memory, input_pad_mask)
            choice = torch.argmax(dec_out, dim=-1)
            outputs = torch.cat([outputs, choice], dim=1)

            # set stop_lens here!
            stop_mask = (choice.squeeze() == EOS_IDX).logical_and(stop_lens == max_len)
            stop_lens[stop_mask] = i
            i +=1
            keep_gen = torch.any(stop_lens.eq(max_len)) and i < max_len

        # Maybe this is a bit overkil...
        # Cut off SOS
        res = outputs[:, 1:]
        pad_mask = sent_lens_to_mask(stop_lens, res.shape[1])
        res = res * pad_mask
        return res, stop_lens

    def decode_sequence(self, tgt, tgt_lens, enc_outputs, masks, teacher_ratio=1):
        input_mask, input_pad_mask = masks
        # Need to adjust tgt with "go" padding:
        sos = torch.as_tensor([SOS_IDX for i in range(0, batch_size)], device=enc_outputs.device, dtype=torch.long).unsqueeze(1)
        tgt_input = torch.cat([sos, tgt[:, :-1, :]], dim=1)

        embedded_tgt, (tgt_mask, tgt_pad_mask) = self.preprocess(tgt_input, tgt_lens, enc=False)
        outs = self.decoder(embedded_tgt, enc_outputs, tgt_mask, None,
                                        tgt_pad_mask, input_pad_mask)

        return self.postprocess(dec_out)

    def forward(self, text, text_len, noise_in=False, teacher_ratio=1):
        enc_outputs, (input_mask, input_pad_mask) = self.encode(mel, mel_len, noise_in)
        return self.decode_sequence(text, text_len, enc_outputs, input_mask, input_pad_mask)
        

class TextRNN(AutoEncoderNet):
    # TODO: Fill in with pre/post needed and enc/dec
    def __init__(self, args):
        super(TextRNN, self).__init__()
        self.prenet = TextPrenet(args.t_emb_dim, args.e_in, p=args.t_pre_drop)
        self.encoder = RNNEncoder(args.e_in, args.hidden, dropout=args.e_drop, 
            num_layers=args.num_layers, bidirectional=args.e_bi)
        self.decoder = RNNDecoder(args.e_in, args.hidden * self.encoder.num_dir, args.hidden, dropout=args.d_drop, 
            num_layers=args.num_layers, attention=args.d_attn, attn_dim=args.attn_dim)
        self.postnet = TextPostnet(args.hidden, args.t_post_drop)

    def preprocess(self, text, text_lens, noise_in=False):
        # raw_input_tensor should be a LongTensor of size [batch_size x seq_len x 1]
        # should already be padded as well
        # Get a mask of 0 for no padding, 1 for padding of size [batch_size x seq_len]
        pad_mask = text.eq(PAD_IDX) 
        embedded_phonemes = self.prenet.emb_dropout(self.prenet.embed(text))
        if noise_in:
            embedded_phonemes = noise_fn(embedded_phonemes)
        return self.prenet.forward_fcn(embedded_phonemes), pad_mask

    def encode(self, text, text_lens, noise_in=False):
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
        embedded_phonemes, pad_mask = self.preprocess(text, text_lens, noise_in)
        enc_output, hidden_state = self.encoder(embedded_phonemes)
        return (hidden_state, enc_output), pad_mask

    def decode_sequence(self, target, tgt_lens, enc_outputs, enc_ctxt_mask, teacher_ratio=1):
        """
        For easier training!  target is teacher forcing [batch_size x seq_len]
        """
        hidden_state, enc_output = enc_outputs
        batch_size, max_out_len = target.shape[0], target.shape[1]
        out_list = []
        input_ = torch.as_tensor([SOS_IDX for i in range(0, batch_size)], device=enc_output.device, dtype=torch.long).unsqueeze(1)
        if self.decoder.attention == "lsa":
            self.decoder.attention_layer.init_memory(enc_output)
        for i in range(max_out_len):
            dec_out, hidden_state = self.decode(input_, hidden_state, enc_output, enc_ctxt_mask)
            out_list.append(dec_out)
            outputs = torch.cat(out_list, dim=1)
            out_list = [outputs]
            if random.random() < teacher_ratio:
                input_ = target[:, 0:i+1]
            else:
                input_ = torch.argmax(outputs, dim=-1)
        if self.decoder.attention == "lsa":
            self.decoder.attention_layer.clear_memory()
        return outputs

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
        input_embed = self.prenet(input_)[:, -1, :].unsqueeze(1)
        dec_output, dec_hidden = self.decoder(input_embed, hidden_state, enc_output, enc_ctxt_mask)
        return self.postprocess(dec_output), dec_hidden

    def postprocess(self, dec_output, distrib=False):
        
        final_out = self.postnet(dec_output)
        if distrib:
            # If we want to return distribution, take log softmax!
            return F.log_softmax(final_out, dim=-1)
        else:
            return final_out

    def infer_sequence(self, enc_outputs, enc_ctxt_mask, max_len=300):

        hidden_state, enc_output = enc_outputs
        batch_size = enc_output.shape[0]
        outputs = torch.as_tensor([SOS_IDX for i in range(0, batch_size)], device=enc_output.device, dtype=torch.long).unsqueeze(1)
        stop_lens = torch.full((batch_size,), max_len, device=enc_output.device)
        
        # get a all SOS for first timestep
        input_ = torch.as_tensor([SOS_IDX for i in range(0, batch_size)], device=enc_output.device, dtype=torch.long).unsqueeze(1)
        i = 0
        keep_gen = torch.any(stop_lens.eq(max_len)) and i < max_len
        if self.decoder.attention == "lsa":
            self.decoder.attention_layer.init_memory(enc_output)
        while keep_gen:
            dec_out, hidden_state = self.decode(outputs, hidden_state, enc_output, enc_ctxt_mask)
            choice = torch.argmax(dec_out, dim=-1)
            outputs = torch.cat([outputs, choice], dim=1)
            # set stop_lens here!

            # double check this!
            stop_mask = (choice.squeeze() == EOS_IDX).logical_and(stop_lens == max_len)
            stop_lens[stop_mask] = i
            i += 1

            keep_gen = torch.any(stop_lens.eq(max_len)) and i < max_len

        # Maybe this is a bit overkil...
        if self.decoder.attention == "lsa":
            self.decoder.attention_layer.clear_memory()
        res = outputs[:, 1:]
        pad_mask = sent_lens_to_mask(stop_lens, res.shape[1])
        res = res * pad_mask
        return res, stop_lens

    def forward(self, text, text_len, noise_in=False, teacher_ratio=1):
        encoder_outputs, pad_mask = self.encode(text, text_len, noise_in=noise_in)
        pred = self.decode_sequence(text, text_len, encoder_outputs, pad_mask, teacher_ratio)
        return pred