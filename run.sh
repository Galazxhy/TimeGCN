#! /bin/bash

source /d/Apps/Anaconda/Anaconda/etc/profile.d/conda.sh

# Overall setting
mode='train'
forec=1
data='./data/Floatation'
model='DSSL'
device='cuda'
batch_size=4
seq_length=60

# Structure learning parameters
SLepoch=20
SLlr=1e-4
SLlatent_dim=64
SLif_img=1

# Predictor parameter
PRElr=1e-3
class_dim=1

# Time-series gragh predictor learning parameters
TGPhidden_dim=256
TGPepoch=400
TGPbeta=1e-4
TGPtau=20
TGPfc=0.2

# LSTM predictor learning parameters
LSTMhidden_dim=64
LSTMnum_layers=2
LSTMbidirec=0

conda activate GNN && python run.py $mode --forec $forec --data $data --model $model --device $device --batch_size $batch_size --seq_length $seq_length --SLepoch $SLepoch --SLlr $SLlr --SLlatent_dim $SLlatent_dim --SLif_img $SLif_img --PRElr $PRElr --class_dim $class_dim --TGPhidden_dim $TGPhidden_dim --TGPepoch $TGPepoch --TGPbeta $TGPbeta --TGPtau $TGPtau --TGPfc $TGPfc --LSTMhidden_dim $LSTMhidden_dim --LSTMnum_layers $LSTMnum_layers --LSTMbidirec $LSTMbidirec