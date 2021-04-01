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
    def __init__(self):
        super(RNNEncoder, self).__init__()

class RNNDecoder(nn.Module):
    # TODO: Write this
    def __init__(self):
        super(RNNDecoder, self).__init__()

