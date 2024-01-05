# -*- coding: utf-8 -*- #
 
# ------------------------------------------------------------------
# File Name:        run
# Author:           郑徐瀚宇
# Version:          ver0_1
# Created:          2023/11/21
# Description:      Run the main program
# Function List:    run() -- run with config
# History:
#       <author>        <version>       <time>      <desc>
#       郑徐瀚宇         ver0_1          2023/11/21  None
# ------------------------------------------------------------------

import os
import torch
from torch import optim
import argparse
from model import model, train, valid, utils
from data import TSdataset
from torch.utils.tensorboard import SummaryWriter

def run(args):
    """Running program
    ---
    Parameters:
        args: Entrance parameters
    """
    print('--------Loading Data--------')
    data = TSdataset.GetData(args.data, args.batch_size, args.seq_length, args.mode)
    print('--------Data Loaded--------')
    if not os.path.exists("./TBlog"):
        os.mkdir("./TBLog")
    i = 0
    while(True):   
        if os.path.exists("./TBLog/exp"+str(i)):
            i = i + 1
        else:
            writer = SummaryWriter('./TBLog/exp'+str(i))
            break
    
    # Train mode
    if args.mode == 'train':
        # With DSSTSL
        if args.model == 'DSSL':
            # DSSLModule
            net = model.DSSLTS(
                data.getInputShape[1], 
                args.SLlatent_dim, 
                args.seq_length,
                args.SLif_img, 
                data.getInputShape[1],
                args.TGPhidden_dim, 
                args.class_dim, 
                args.forec,
                args.TGPbeta,
                args.TGPtau,
                args.TGPfc
            )
            net.to(args.device)
            # optimizer
            SLopt = optim.Adam(net.sslParam, lr=args.SLlr, weight_decay=1e-5)
            PREopt = optim.Adam(net.predParam, lr=args.PRElr, weight_decay=1e-5)
            # train
            if args.forec:
                print('--------Training Forecasting Task--------')
            else:
                print('--------Training Classification Task--------')
            rec_loss = train.strcutureLearning(net, SLopt, data.getTrainDataloader, args, writer)
            pre_loss = train.perdictorLearning(net, PREopt, data.getTrainDataloader, data.getTestDataloader, args, writer)
            # valid
            print('--------Validating--------')
            valid_acc, mae, rmse, r2_score = valid.valid(net, data.getTestDataloader, args)
            if args.forec:
                utils.print_log(i, args, {'Reconstruction loss':rec_loss, 'Prediction loss':pre_loss, 'Test MAE': mae, 'Test RMSE':rmse, 'Test R2':r2_score})
            else:
                utils.print_log(i, args, {'Reconstruction loss':rec_loss, 'Prediction loss':pre_loss, 'Test Accuracy':valid_acc})
            
        elif args.model == 'LSTM':
            # LSTM baseline
            net = model.TSLSTM(
                data.getInputShape[1],
                args.LSTMhidden_dim, 
                args.class_dim, 
                args.LSTMnum_layers,
                args.LSTMbidirec
            ).to(args.device)
            # optimizer
            LSTMopt = optim.Adam(net.parameters(), lr=args.PRElr, weight_decay=1e-5)
            # train
            if args.forec:
                print('--------Training Forecasting Task--------')
            else:
                print('--------Training Classification Task--------')
            pre_loss = train.perdictorLearning(net, LSTMopt, data.getTrainDataloader, args, writer)
            # valid
            print('--------Validating--------')
            valid_acc = valid.valid(net, data.getTestDataloader, args)
            utils.print_log(i, args, {'Prediction loss':pre_loss, 'Testing Accuracy':valid_acc})
    # Test mode
    elif args.mode == 'test':
        print('--------Testing--------')
        net = torch.load('./save/xx.pth')
        valid_acc = valid.valid(net, data.getTestDataloader, args)
        utils.print_log(args, {'Testing Accuracy': valid_acc})
    else:
        print('Running Mode Error!')
        return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Running Parameters')
    # Overall setting
    parser.add_argument('mode', type=str, default='train', help='Running code with train mode or not')
    parser.add_argument('--forec', type=int, default=0, help='Choosing forecasting task of classification task')
    parser.add_argument('--data', type=str, default='./data/Floatation', help='Root dir of dataset')
    parser.add_argument('--model', type=str, default='DSSL', help='Choosing model from: [DSSL, LSTM]')
    parser.add_argument('--device', type=str, default='cuda', help='Calculating device')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size of structure learning')
    parser.add_argument('--seq_length', type=int, default=-1, help='Length of sampling sequence length when forcasting (-1 for classification)')
    # Strcuture learning parameters
    parser.add_argument('--SLepoch', type=int, default=100, help='Structure learning epoch')
    parser.add_argument('--SLlr', type=float, default=0.001, help='Structure learning rate')
    parser.add_argument('--SLlatent_dim', type=int, default=10, help='Number of latent variables')
    parser.add_argument('--SLif_img', type=int, default=0, help='If data is of image type')
    # Predictor parameters
    parser.add_argument('--PRElr', type=float, default=0.001, help='Predictor learning rate')
    parser.add_argument('--class_dim', type=int, default=1, help='Number of classes (1 for forecasting task)')
    # Time-series graph predictor learning parameters
    parser.add_argument('--TGPhidden_dim',type=int, default=128, help='Number of time-series graph predictor hidden layer')
    parser.add_argument('--TGPepoch', type=int, default=100, help='Predictor training epcoch')
    parser.add_argument('--TGPbeta', type=float, default=0.1, help='Hidden structure forecasting coefficient')
    parser.add_argument('--TGPtau', type=int, help='Time message passing length')
    parser.add_argument('--TGPfc', type=float, help='Forgetting Coefficient')
    # LSTM predictor learning parameters
    parser.add_argument('--LSTMhidden_dim', type=int, default=128, help='Number of hidden layer of LSTM')
    parser.add_argument('--LSTMnum_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--LSTMbidirec', type=int, default=0, help='If using bidirectional LSTM')
    args = parser.parse_args()

    run(args)
