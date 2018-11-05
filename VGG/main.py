import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import argparse
import torch.utils.data
from VGG.model import VGG16
import os

parser = argparse.ArgumentParser()
parser.add_argument('--num_bases', type=int, default=5, help="number of bases")
parser.add_argument('--type_bases', default='FB', help="type of bases, support FB, random, PCA")
parser.add_argument('--bases_path', default='bases_VGG16.npy', help='path for type_bases=PCA')
parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
parser.add_argument('--lr', type=float, default=0.05, help='learning rate for training')
parser.add_argument('--save_dir', default='save_tmp', help='save path for model')
parser.add_argument('--data_dir', default='/Users/hzc/dataset/cifar10', help='dataset path for cifar10')
parser.add_argument('--epochs', type=int, default=300, help='number of epochs for training')
parser.add_argument('--weight_decay', type=float, default=5e-4, help="weight decay for the model")
args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

assert args.type_bases in ['FB', 'random', 'PCA']
if args.type_bases == 'PCA':
    bases = np.load(args.bases_path)
else:
    bases = args.type_bases

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root=args.data_dir, train=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ]), download=True),
    batch_size=args.batch_size, shuffle=True,
    num_workers=4, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root=args.data_dir, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]), download=True),
    batch_size=args.batch_size, shuffle=False,
    num_workers=4, pin_memory=True)

model = VGG16(args.num_bases, bases)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

for epoch in range(args.epochs):
    scheduler.step()
    model.train()
    print("running epoch {}".format(epoch+1))
    print("-"*15)
    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)
        logits, preds = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            acc = torch.sum(labels==preds)
            print("epoch {} batch {}, training loss={:5f} accuracy={:4f}".format(epoch+1, i+1, loss, acc))

    model.eval()
    sum_acc = 0.0
    sum_loss = 0.0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        logits, preds = model(images)
        sum_acc +=torch.sum(labels==preds)
        sum_loss += criterion(logits, labels)

    print("epoch {} , testing loss={:5f} accuracy={:4f}".format(epoch + 1, sum_loss/len(test_loader), sum_acc/len(test_loader)))

    if (epoch+1) % 5 == 0:
        state_dict = model.state_dict()
        torch.save(state_dict, args.save_dir)


