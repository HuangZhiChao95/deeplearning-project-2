from __future__ import print_function
import sys
sys.path.append('./models/')
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from models.resnet_dcf import resnet50

def train(model, device, train_loader, optimizer, epoch):
    # switch to training mode
    model.train()
    correct = 0
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(F.log_softmax(output), target)
        loss.backward()
        optimizer.step()   
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader, mode):
    # switch to evluation mode
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(F.log_softmax(output), target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc =  correct / len(test_loader.dataset)
    print('\n', mode, 'set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * test_acc))
    return test_loss, test_acc


def main(model, batch_size, lr, momentum, epochs, weight_decay,step_size, gamma, no_cuda=True):
    # cpu or gpu
    use_cuda = not no_cuda and torch.cuda.is_available()
    print('use_cude:', use_cuda)
    torch.manual_seed(seed)
    device = torch.device("cuda:1" if use_cuda else "cpu")
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    model.to(device)
    
    # download data
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../cifar10', train=True, download=True,
                   transform=transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])),
                batch_size=batch_size, shuffle=True, **kwargs)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../cifar10', train=False, download=True,
                        transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    
    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    for epoch in range(1, epochs + 1):    
        scheduler.step()
        train(model, device, train_loader, optimizer, epoch)
        tr_loss, tr_acc = test(model, device, train_loader, 'train') 
        te_loss, te_acc = test(model, device, test_loader, 'test')  
        loss[epoch-1,0] = tr_loss
        loss[epoch-1,1] = te_loss
        accuracy[epoch-1,0] = tr_acc
        accuracy[epoch-1,1] = te_acc
        log = pd.DataFrame(columns=['loss_train','loss_val','acc_train','acc_test'])
        log['loss_train'] = loss[:,0]
        log['loss_val'] = loss[:,1]
        log['acc_train'] = accuracy[:,0]
        log['acc_test'] = accuracy[:,1]
        log.to_csv('log.csv',index=False)
        
def plot(loss, accuracy):
    plt.figure(figsize=(8,10))
    
    plt.subplot(2,1,1)
    plt.plot(loss[:,0], 'o-')
    plt.plot(loss[:,1], '.-')
    plt.legend(['train', 'test'])
    plt.xlabel('epoch')
    plt.title('loss')
    
    plt.subplot(2,1,2)
    plt.plot(accuracy[:,0], 'o-')
    plt.plot(accuracy[:,1], '.-')
    plt.legend(['train', 'test'])
    plt.title('accuracy')
    plt.xlabel('epoch')

if __name__ == '__main__':
    seed = 1
    log_interval = 50
    in_channels  = 3

    epochs = 400
    loss = np.zeros((epochs,2))
    accuracy = np.zeros((epochs,2))

    kwargs={"in_channels":3,
            "num_bases":9,
            "initializer":'PCA'}

    model = resnet50(**kwargs)
    main(model, 
        batch_size=64, 
        lr=1e-1, 
        momentum=0.9, 
        epochs=epochs, 
        weight_decay=5e-4, 
        step_size=30,
        gamma=0.5,
        no_cuda=False)
            