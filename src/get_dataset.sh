#!/bin/bash

set -ex

# Create the data folder
mkdir -p data
cd data

# Download the LJSpeech dataset into the data folder and splits
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xvf LJSpeech-1.1.tar.bz2
cp -r ./splits/. ./LJSpeech-1.1/

cd ../
python prepare_data.py
