# -*- coding: utf-8 -*- #
 
# ------------------------------------------------------------------
# File Name:        valid
# Author:           郑徐瀚宇
# Version:          ver0_1
# Created:          2023/11/21
# Description:      Validation function
# Function List:    valid() -- validate model
# History:
#       <author>        <version>       <time>      <desc>
#       郑徐瀚宇         ver0_1          2023/11/21  None
# ------------------------------------------------------------------
from model import utils
import torch
from tqdm import tqdm
import torcheval.metrics.functional as F

def valid(model, validDataloader, args):
    """Validating perdictor
    ---
    Parameters:
        model: Predictor
        validDataloader: Validation Dataloadeer for Structure Learning (Dataloader)
    ---
    Return:
        accuracy:
    """
    # tbar = tqdm(total=len(validDataloader))
    # tbar.set_description(f'Validating Model')
    accuracy = 0.0
    mae = 0.0
    rmse = 0.0
    r2_score = 0.0
    yAll = None
    ySLAll = None
    outAll = None
    maskAll = None
    with torch.no_grad():
        for data in validDataloader:
            # data: (x: [batch_size, seq_length, num_feature], ySL: [batch_size, seq_length, num_class]) y:[batch_size, num_class]
            x, ySL, mask, y = data
            x = x.to(args.device)
            ySL = ySL.to(args.device)
            mask = mask.to(args.device)
            y = y.to(args.device)
            mae_fn = torch.nn.L1Loss()
            output = model.predictor(x) # [batch_size, seq_length, 1]
            outAll = utils.ts_append(outAll, output)
            maskAll = utils.ts_append(maskAll, mask)
            ySLAll = utils.ts_append(ySLAll, ySL)
            yAll = utils.ts_append(yAll, y)
        outAll = utils.lb_denormalize(outAll)
        ySLAll = utils.lb_denormalize(ySLAll)
        yAll = utils.lb_denormalize(yAll)
        if args.forec:
            if args.model == 'DSSL':
                mae = mae_fn(torch.masked_select(outAll, maskAll), torch.masked_select(ySLAll, maskAll)).item()
                rmse = torch.sqrt(F.mean_squared_error(torch.masked_select(ySLAll, maskAll), torch.masked_select(outAll, maskAll))).item()
                print(torch.masked_select(ySLAll, maskAll), torch.masked_select(outAll, maskAll))
                r2_score = F.r2_score(torch.masked_select(outAll, maskAll), torch.masked_select(ySLAll, maskAll)).item()
            else:
                accuracy = mae_fn(outAll, ySLALL)
        else:
            if args.model == 'DSSL':
                outAll = outAll.argmax(dim=1)
                accuracy = torch.eq(outAll, yAll).sum(dim=0) / outAll.shape[0]
            else:
                outAll = outpAll.argmax(dim=1)
                accuracy = torch.eq(outAll, yAll).sum(dim=0) / outAll.shape[0]
        
            # tbar.update()
    
    # if args.forec:
    #     print(f'Prediction R2 Score:{accuracy/len(validDataloader)}')
    # else:
    #     print(f'Classification Accuracy:{accuracy/len(validDataloader)}')
    # tbar.close()

    return accuracy, mae, rmse, r2_score

