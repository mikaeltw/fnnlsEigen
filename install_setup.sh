#!/bin/bash -ex

rm -rf env
rm -f requirements.txt

python3 -m venv env

. env/bin/activate

python3 -m pip install -U pip==21.1.2

python3 -m pip install pip-tools==6.1.0

pip-compile

python -m pip install -r requirements.txt
