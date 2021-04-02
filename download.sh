#!/bin/bash

set -ex

# Create the data folder
mkdir -p data
cd data

# Download the LJSpeech dataset into the data folder
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xvf LJSpeech-1.1.tar.bz2
rm LJSpeech-1.1.tar.bz2
