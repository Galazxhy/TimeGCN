#! /bin/bash

source /d/Apps/Anaconda/Anaconda/etc/profile.d/conda.sh

# Overall setting
mode='train'
forec=1
data='./data/Floatation'
model='TIGCN'
device='cuda'
batch_size=4
seq_length=60

# Structure learning parameters
SLepoch=20
SLlr=1e-4
SLlatent_dim=64
SLif_img=1

# Predictor parameter 
PRElr=5e-4
class_dim=1

# Time-series gragh predictor learning parameters
TGPhidden_dim=128
TGPepoch=150
TGPbeta=1
TGPtau=22
TGPfc=0.15

# LSTM predictor learning parameters
LSTMhidden_dim=64
LSTMnum_layers=2
LSTMbidirec=0

conda activate GNN && python run.py $mode --forec $forec --data $data --model $model --device $device --batch_size $batch_size --seq_length $seq_length --SLepoch $SLepoch --SLlr $SLlr --SLlatent_dim $SLlatent_dim --SLif_img $SLif_img --PRElr $PRElr --class_dim $class_dim --TGPhidden_dim $TGPhidden_dim --TGPepoch $TGPepoch --TGPbeta $TGPbeta --TGPtau $TGPtau --TGPfc $TGPfc --LSTMhidden_dim $LSTMhidden_dim --LSTMnum_layers $LSTMnum_layers --LSTMbidirec $LSTMbidirec