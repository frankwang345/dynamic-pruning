from torchvision import transforms, datasets
import torch.nn.functional as F
import torch
import argparse
import os

import models
import misc

print = misc.logger.info

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--arch', '-a', default='resnet56', type=str)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--mm', default=0.9, type=float)
parser.add_argument('--wd', default=1e-4, type=float)
parser.add_argument('--epochs', default=160, type=int)
parser.add_argument('--log_interval', default=100, type=int)
parser.add_argument('--train_batch_size', default=128, type=int)

args = parser.parse_args()

args.num_classes = 10 if args.dataset == 'cifar10' else 100

args.device = 'cuda'
torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

args.logdir = 'pretrained/%s/%s' % (args.dataset, args.arch)
misc.prepare_logging(args)

print('==> Preparing data..')

if args.dataset == 'cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

elif args.dataset == 'cifar100':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR100(root='./data/cifar100', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=2)

    testset = datasets.CIFAR100(root='./data/cifar100', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

print('==> Initializing model...')
if args.dataset in ['cifar10', 'cifar100']:
    model = models.__dict__['cifar_' + args.arch](args.num_classes)

model = model.to(args.device)

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mm, weight_decay=args.wd)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)


def train(epoch):
    model.train()
    for i, (data, target) in enumerate(trainloader):
        data = data.to(args.device)
        target = target.to(args.device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        pred = output.max(1)[1]
        acc = (pred == target).float().mean()

        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}, Accuracy: {:.4f}'.format(
                epoch, i, len(trainloader), loss.item(), acc.item()
            ))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1)[1]
            correct += (pred == target).float().sum().item()

    test_loss /= len(testloader.dataset)
    acc = correct / len(testloader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}\n'.format(
        test_loss, acc
    ))
    return acc

for epoch in range(args.epochs):
    scheduler.step()
    train(epoch)
    acc = test()

    torch.save(model.state_dict(), os.path.join(args.logdir, 'checkpoint.pth'))

print('Final saved model test accuracy = %.4f' % acc)
