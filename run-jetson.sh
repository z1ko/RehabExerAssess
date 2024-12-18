#! /bin/bash
python3.8 train.py --model gcn --Pnorm --epoch 200 --batch_size 16
python3.8 train.py --model ri-gcn --Pnorm --epoch 200 --batch_size 16
python3.8 train.py --model va-gcn --Pnorm --epoch 200 --batch_size 16