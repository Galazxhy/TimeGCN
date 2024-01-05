# -*- coding: utf-8 -*- #
 
# ------------------------------------------------------------------
# File Name:        train
# Author:           郑徐瀚宇
# Version:          ver0_1
# Created:          2023/11/21
# Description:      Train VAE & GNNPredictor
# Function List:    strcutureLearning() -- Train VAE for Graph structure of time-series
#                   perdictorLearning() -- Train Timg-GCN
# History:
#       <author>        <version>       <time>      <desc>
#       郑徐瀚宇         ver0_1          2023/11/21  None
# ------------------------------------------------------------------

import torch
import time
from tqdm import tqdm
import torch.nn.functional as F
from model import valid


def strcutureLearning(model, opt, TSdataloader, args, writer):
    """Learning Graph Structure
    ---
    Parameters:
        model: DSSTSL Structure Learning Module
        opt: Optimizer
        TSdata: Time Series Data for Structure Learning (Dataloader)
        args: Training Parameters
        writer:
    ---
    Return:
        total_loss
    """
    tbar = tqdm(total=args.SLepoch, position=0, leave=True)
    step = 0

    for i in range(args.SLepoch):
        tbar.set_description(f'Structure Learning Epoch {i+1} / {args.SLepoch}')
        total_loss = 0.0
        for data in TSdataloader:
            # data: (x: [batch_size, num_feature])
            # t1 = time.time()
            x, _, _, _ = data
            x = x.to(args.device)
            
            opt.zero_grad()
            loss = model.ssl(x)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            writer.add_scalar('Rec Loss', loss.item(), step)
            step += 1
            # t2 = time.time()
            # print(t2-t1)
        
        total_loss = total_loss / len(TSdataloader)
        
        tbar.set_postfix(SLloss='{:.5f}'.format(total_loss))
        tbar.update()
    tbar.close()

    return total_loss
    

def perdictorLearning(model, opt, TSdataloader, valid_dataloader, args, writer):
    """Learning perdictor
    ---
    Parameters:
        model: Predictor
        opt: Optimizer
        TSdataloader: Time Series Data for Structure Learning (Dataloader)
        args: Training Parameters
        writer:
    ---
    Return:
        total_loss
    """
    tbar = tqdm(total=args.TGPepoch, position=0, leave=True)

    for i in range(args.TGPepoch):
        tbar.set_description(f'Predictor Learning Epoch {i + 1} / {args.TGPepoch}')
        total_loss = 0.0
    
        for data in TSdataloader:
            # data: (x: [batch_size, num_feature], ySL: [batch_size, 2, seq_length, num_class])
            x, ySL, mask, y  = data
            x = x.to(args.device)
            ySL = ySL.to(args.device)
            mask = mask.to(args.device)
            y = y.to(args.device)
            if args.forec:
                # Forcasting task
                opt.zero_grad()
                if args.model == 'DSSL':
                    output = model.predictor(x) # [batch_size, seq_length, num_class
                    loss = F.mse_loss(torch.masked_select(output, mask), torch.masked_select(ySL, mask))
                else:
                    output = model(x) # [batch_size, num_class]
                    loss = F.mse_loss(output, y)
                loss.backward(retain_graph=True)
                opt.step()
                
                total_loss += loss.item()
            else:
                # Classification task
                opt.zero_grad()
                if args.model == 'DSSL':
                    output = model.predictor(x) # [batch_size, num_class]
                else:
                    output = model(x) # [batch_size, num_class]
                loss = F.cross_entropy(output, y.long())
                loss.backward()
                opt.step()
                total_loss += loss.item()
        
        total_loss = total_loss / len(TSdataloader)
        writer.add_scalar('Traning MSE Loss', total_loss, i)
        acc, mae, rmse, r2_score = valid.valid(model, valid_dataloader, args)
        if args.forec:
            writer.add_scalar('Test MAE', mae, i)
            writer.add_scalar('Test RMSE', rmse, i)
            writer.add_scalar('Test R2 Score', r2_score, i)
        else:
            writer.add_scalar('Accuracy', acc, i)
        tbar.set_postfix(Preloss='{:.5f}'.format(total_loss))
        tbar.update()

    tbar.close()

    return total_loss
