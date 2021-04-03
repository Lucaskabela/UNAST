'''
Contains the nn.Modules which compose the networks. This includes modules 
for preprocessing speech and text, the transformer & RNN encoder/decoder, & post
processing modules for text and speech.
'''
# TODO: Consider adding pre/post net base class.

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
        self.hid2out = nn.Linear(self.num_dir * hidden, d_out )
    
    def forward(self, sequence):
        output, (h, c) = self.rnn(sequence)
        return self.hid2out(output)

class RNNDecoder(nn.Module):
    def __init__(self, encoder_out_size, d_in, hidden, d_out, dropout=.2, num_layers=1, bidirectional=False, attention=False):
        super(RNNDecoder, self).__init__()
        self.attention = attention
        if self.attention:
            # since encoder output size should be d_in
            self.input_size = encoder_out_size + d_in
        else:
            self.input_size = d_in
        self.rnn = nn.LSTM(self.input_size, hidden, num_layers=num_layers, 
            bidirectional=bidirectional, batch_first=True, dropout=dropout)
        
        # Luong attention
        if self.attention:
            self.attention_layer = nn.Linear(hidden, encoder_out_size)
            self.out_layer = nn.Linear(hidden + encoder_out_size, d_out)
            self.attn_softmax = nn.Softmax(dim=1)
        else:
            self.out_layer = nn.Linear(hidden, d_out)
    
    def forward(self, latent_embed, state, enc_output, enc_ctxt_mask):
        # TODO: Check shape here?
        rnn_in = latent_emb.unsqueeze(0)
        output, hidden = self.rnn(rnn_in, state)
        if self.attention:
            # TODO: attention computation! 
            pass
        return self.out_layer(output), hidden