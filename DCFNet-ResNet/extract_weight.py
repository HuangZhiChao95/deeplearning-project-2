import torch
import numpy as np
pre_weight = torch.load('./pretrained_weight/resnet50.pth')
keys = list(pre_weight.keys())
weights = []
for layer in keys:
    if 'conv2' in layer or layer=='conv1.weight':
        weight = pre_weight[layer].detach().numpy()
        weight = weight.transpose((0,1,2,3))
        weights.append(weight)
np.save('./bases/bases_resnet.npy', weights)
        
        


