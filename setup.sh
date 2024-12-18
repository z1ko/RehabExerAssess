#! /bin/bash -i

conda create -n RehabExerAccess python=3.7.11
conda activate RehabExerAccess

pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111

pip install -r requirements.txt