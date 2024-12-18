#! /bin/bash
python3 train.py --model gcn --Pnorm --epoch 2
python3 train.py --model ri-gcn --Pnorm --epoch 200
python3 train.py --model va-gcn --Pnorm --epoch 200