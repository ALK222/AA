#!/usr/bin/env bash

mkdir ./src

cd ./src
python3 -m venv .venv

source .venv/bin/activate

pip install pipreqs

pip install -r ./requirements.txt