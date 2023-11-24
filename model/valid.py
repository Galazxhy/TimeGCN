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
    with torch.no_grad():
        for data in validDataloader:
            # data: (x: [batch_size, seq_length, num_feature], ySL: [batch_size, seq_length, num_class]) y:[batch_size, num_class]
            x, ySL, y = data
            x = x.to(args.device)
            ySL = ySL.to(args.device)
            y = y.to(args.device)
            if args.forec:
                if args.model == 'DSSL':
                    output = model.predictor(x) # [batch_size, seq_length, 1]
                    accuracy += F.r2_score(output[ySL[:, 1] == 1], ySL[:, 0][ySL[:, 1] == 1])
                else:
                    output = model(x) # [batch_size, 1]
                    accuracy += F.r2_score(output)
            else:
                if args.model == 'DSSL':
                    output = model.predictor(x) # [batch_size, class_dim]
                    output = output.argmax(dim=1)
                    accuracy += torch.eq(output, y).sum(dim=0) / output.shape[0]
                else:
                    output = model(x) # [batch_size, class_dim]
                    output = output.argmax(dim=1)
                    accuracy += torch.eq(output, y).sum(dim=0) / output.shape[0]
        
            # tbar.update()
    
    # if args.forec:
    #     print(f'Prediction R2 Score:{accuracy/len(validDataloader)}')
    # else:
    #     print(f'Classification Accuracy:{accuracy/len(validDataloader)}')
    # tbar.close()

    return accuracy.item() / len(validDataloader)
