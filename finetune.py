from torchvision import transforms, datasets
import torch.nn.functional as F
import numpy as np
import torch
import argparse
import os

from decision import default_graph, apply_func, replace_func, set_deterministic_value
import models
import misc

print = misc.logger.info

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--arch', '-a', default='resnet56', type=str)
parser.add_argument('--action_num', default=40, type=int)
parser.add_argument('--sparsity_level', default=0.7, type=float)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--epochs', default=160, type=int)
parser.add_argument('--log_interval', default=100, type=int)
parser.add_argument('--train_batch_size', default=128, type=int)

args = parser.parse_args()

args.num_classes = 10 if args.dataset == 'cifar10' else 100

args.device = 'cuda'
torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

args.logdir = 'finetune-decision-%d/%s-%s/sparsity-%.2f' % (
    args.action_num, args.dataset, args.arch, args.sparsity_level
)
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
model = models.__dict__['cifar_' + args.arch](args.num_classes)
model_params = []
for p in model.parameters():
    model_params.append(p)

if args.arch in ['resnet20', 'resnet56']:
    from decision import init_decision_basicblock, decision_basicblock_forward
    init_func = init_decision_basicblock
    new_forward = decision_basicblock_forward
    module_type = 'BasicBlock'

else:
    from decision import init_decision_convbn, decision_convbn_forward
    init_func = init_decision_convbn
    new_forward = decision_convbn_forward
    module_type = 'ConvBNReLU'

print('==> Transforming model...')

apply_func(model, module_type, init_func, action_num=args.action_num)
replace_func(model, module_type, new_forward)

print('==> Loading pretrained decision model...')
ckpt = torch.load(
    'logs/decision-%d/%s-%s/sparsity-%.2f/model.pth.tar' % (
        args.action_num, args.dataset, args.arch, args.sparsity_level
))
model.load_state_dict(ckpt['state_dict'])

model = model.to(args.device)

optimizer = torch.optim.SGD(model_params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)

def train(epoch):
    model.train()
    apply_func(model, 'DecisionHead', set_deterministic_value, deterministic=True)
    for i, (data, target) in enumerate(trainloader):
        default_graph.clear_all_tensors()

        data = data.to(args.device)
        target = target.to(args.device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            selected_channels = default_graph.get_tensor_list('selected_channels')
            concat_channels = torch.cat(selected_channels, dim=1)
            sparsity = (concat_channels != 0).float().mean()
            acc = (output.max(1)[1] == target).float().mean()

            print('Train Epoch: %d [%d/%d]\tLoss: %.4f, '
                  'Sparsity: %.4f, Accuracy: %.4f' % (
                epoch, i, len(trainloader), loss.item(),
                sparsity.item(), acc.item()
            ))

def test():
    model.eval()
    apply_func(model, 'DecisionHead', set_deterministic_value, deterministic=True)
    test_loss = []
    test_sparsity = []
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            default_graph.clear_all_tensors()

            data, target = data.to(args.device), target.to(args.device)
            output = model(data)

            selected_channels = default_graph.get_tensor_list('selected_channels')
            concat_channels = torch.cat(selected_channels, dim=1)

            test_loss.append(F.cross_entropy(output, target).item())
            test_sparsity.append((concat_channels != 0).float().mean().item())

            pred = output.max(1)[1]
            correct += (pred == target).float().sum().item()

    acc = correct / len(testloader.dataset)
    print('Test set: Loss: %.4f, '
          'Sparsity: %.4f, Accuracy: %.4f\n' % (
        np.mean(test_loss), np.mean(test_sparsity), acc
    ))
    return acc, np.mean(test_sparsity)


for epoch in range(args.epochs):
    train(epoch)
    acc, sparsity = test()
    torch.save(model.state_dict(), os.path.join(args.logdir, 'checkpoint.pth'))
    print('Save checkpoint @ Epoch %d, Accuracy = %.4f, Sparsity = %.4f\n' % (
        epoch, acc, sparsity
    ))
