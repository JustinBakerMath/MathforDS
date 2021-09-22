#!/bin/sh

cd ./Project1
python data/generator.py
python main.py
read -p "$*"
