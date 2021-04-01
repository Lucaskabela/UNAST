#-*- coding: utf-8 -*-


'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run
through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details.
'''
from  data import cmudict

_pad        = '_'
_eos        = '~'
_space = ' '
# phonemes gotten from the cmu dict / eng-to-ipa package
_phonemes = ['ˈ', 'a', 'b', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'z', 'æ', 'ð', 'ŋ', 'ɑ', 'ɔ', 'ə', 'ɛ', 'ɪ', 'ʃ', 'ʊ', 'ʒ', 'ʤ', 'ʧ', 'θ']
# some words (OOV, mispelled, slang) we don't know how to transcribe into
# phonemes, so we pass it through unchanged. We need to add those letters into
# our vocab
_missing_chars = ['c', 'q', 'x', 'y']
# we have a special char to denote when we fail to transcribe
_special_char = '*'

# Export all symbols:
symbols = [_pad, _eos, _space, _special_char] + _phonemes + _missing_chars


if __name__ == '__main__':
    print(symbols)