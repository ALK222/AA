#!/usr/bin/env bash

mkdir ./src

cd ./src
python3.11 -m venv .venv

source .venv/bin/activate

pip install pipreqs

pip install -r ./requirements.txt