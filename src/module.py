'''
Contains the nn.Modules which compose the networks. This includes modules 
for preprocessing speech and text, the transformer & RNN encoder/decoder, & post
processing modules for text and speech.
'''
# TODO: Consider adding pre/post net base class.
import torch.nn as nn

class SpeechPrenet(nn.Module):
    # TODO: Fill in from TTS repo :) 
    def __init__(self):
        super(SpeechPrenet, self).__init__()

class SpeechPostnet(nn.Module):
    # TODO: Fill in from TTS repo :) 
    def __init__(self):
        super(SpeechPostnet, self).__init__()

class TextPrenet(nn.Module):
    # TODO: Fill in from TTS repo :) 
    def __init__(self):
        super(TextPrenet, self).__init__()

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
    def __init__(self, d_in, hidden, d_out, dropout=.2, num_layers=1, bidirectional=False):
        super(RNNEncoder, self).__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        self.num_dir = 2 if bidirectional else 1
        
        # TODO: expirement with something else than LSTM
        self.rnn = nn.LSTM(d_in, hidden, num_layers=num_layers, 
            bidirectional=bidirectional, batch_first=True, dropout=dropout)
        
        # Consider using this?
        self.hid2out = nn.Linear(self.num_dir * hidden, d_out)
    
    def forward(self, sequence):
        output, (h, c) = self.rnn(sequence)
        return output, (h, c)

class RNNDecoder(nn.Module):
    def __init__(self, enc_out_size, d_in, hidden, d_out, dropout=.2, num_layers=1, attention=False):
        super(RNNDecoder, self).__init__()

        self.attention = attention
        if self.attention:
            self.input_size = enc_out_size + d_in
        else:
            self.input_size = d_in

        self.rnn = nn.LSTM(self.input_size, hidden, num_layers=num_layers, 
            batch_first=True, dropout=dropout)
        
        # Luong attention TODO: ADD DROPOUT!?
        if self.attention:
            self.attention_layer = LuongGeneralAttention(hidden_size, enc_out_size)

        self.out_layer = nn.Linear(hidden, d_out)

    def forward(self, embed_decode, hidden_state, enc_output, enc_ctxt_mask):
        # TODO: Check shape here?
        if self.attention:
            attn_W = self.attention_layer(hidden_state[0], enc_output, enc_ctxt_mask)
            decode_input = torch.cat((embed_decode, attn_W), dim=-1)
        else:
            decode_input = embed_decode

        output, hidden = self.rnn(decode_input, hidden_state)
        
        return self.out_layer(output), hidden

class LuongGeneralAttention(nn.Module):
    def __init__(self, hidden_size, enc_out_size):
        self.hidden_size = hidden_size
        self.enc_out_size = enc_out_size
        self.fc1 = nn.Linear(hidden_size + enc_out_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, 1, bias=False)

    def forward(hidden, enc_output, enc_ctxt_mask):
        '''
        Returns the alignment weights
        '''
        src_len = enc_output.shape[0]
        hidden = hidden.repeat(src_len, 1, 1)
        e_o = enc_output.permute(1, 0, 2)
        combined = torch.cat((hidden, e_o), dim=-1)

        # combined is [seq_len x batch_size x hidden + enc_out]
        align_scores = self.fc2(torch.tanh(self.fc1(combined))).squeeze(-1)
        # align_scores is [seq_len x batch_size], so flip and mask all padding
        align_scores = align_scores.permute(1, 0)
        align_scores = align_scores.mask_fill(enc_ctxt_mask==1, -np.inf)

        align_weights = F.softmax(align_scores, dim=-1).unsqueeze(1)
        # align_weights is [batch_size x seq_len x 1] where each entry is score over sequence


        # Note: If input is a (b×n×m) tensor, mat2 is a (b×m×p) tensor
        #  --- out will be a (b×n×p) tensor.
        # And we want a [batch_size x 1 x enc_out_size], so we have the right order
        ctxt = torch.bmm(align_weights, enc_output)
        return ctxt