import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd
from scipy import interpolate
from PIL import Image


def visualizaiton(image, num_layer, num_feature=None, save_name=None):
    net = nn.Sequential(*list(model.children())[:num_layer])
    output = net(image).detach().numpy()[0]
    M,K,_ = output.shape
    M = 16
    print('M =%d, K=%d'%(M,K))
    
    if num_feature is None:
        width = int(M**0.5)
        f = plt.figure(figsize=(4,16))
        for i in range(width):
            for j in range(width):
                plt.subplot(16, width, i*width+j+1)
                feature_map = output[i*width+j]
                feature_map -= feature_map.min()
                feature_map /= (feature_map.max()+1e-5)
                plt.subplots_adjust(left=0, bottom=None, right=None, top=None,
                    wspace=None, hspace=None)
                sns.heatmap(feature_map, cmap=plt.cm.Greys, 
                    cbar=False, xticklabels=False, yticklabels=False,
                    square=True)
    else:
        feature_map = output[num_feature]
        feature_map -= feature_map.min()
        feature_map /= feature_map.max()
        f = plt.figure(figsize=(2,4))
        sns.heatmap(feature_map, cmap=plt.cm.Greys, 
                    cbar=False, xticklabels=False, yticklabels=False,
                    square=True)
    if save_name:
        f.savefig(save_name)  


if __name__ == '__main__':
    
    model = torch.load('./pretrained_weight/model_mix.pkl', map_location='cpu') 
    model.eval()   
    
    # for resnet50 ,conv layers = 0(DCF), 2(ReLu), 4, 5, 6, 7
    # one should manually set expansion to reshape the feature map to a predefined shape
    expansion = 8
    num_layer = 7
    num_feature = None
    save_name = './feature_map/layer_%d_feature%s.png'%(num_layer,str(num_feature))

    
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./cifar10', train=True, download=True,
                   transform=transforms.Compose([
                        transforms.Resize((32*expansion,32*expansion)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])),
                batch_size=1, shuffle=False)
    iterator= train_loader.__iter__()
    image, target = iterator.next()   
    
    visualizaiton(image, num_layer, num_feature, save_name)

    # crop the image
    box = (0,130,270,360)
    img = Image.open(save_name).convert('LA').crop((0,130,270,360)).save(save_name)
    


