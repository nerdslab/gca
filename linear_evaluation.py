import torch
import torchvision
from torch import nn

import torchvision.transforms as transforms
import numpy as np
from cl_models.modules import get_resnet
import hydra
from omegaconf import DictConfig
import logging
import numpy as np

from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, SubsetRandomSampler

from cl_models.modules import get_resnet
from cl_models.linear_model import LinModel

from tqdm import tqdm
from omegaconf import OmegaConf
from torchvision.datasets import ImageFolder
from utils import AverageMeter, load_and_merge_npy_files
import torch.nn.functional as F
from ssl_pretrain import ContrastiveLearning
logger = logging.getLogger(__name__)

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from PIL import Image

from cl_models.simclr import SimCLR, Brightness, StrongCrop
from ssl_pretrain import get_color_distortion

class CIFAR10CDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(label, dtype=torch.long)
        return img, label




def run_epoch(model, dataloader, epoch, optimizer=None, scheduler=None):
    if optimizer:
        model.train()
    else:
        model.eval()

    loss_meter = AverageMeter('loss')
    acc_meter = AverageMeter('acc')
    loader_bar = tqdm(dataloader)
    for x, y in loader_bar:
        x, y = x.cuda(), y.cuda()
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

        acc = (logits.argmax(dim=1) == y).float().mean()
        loss_meter.update(loss.item(), x.size(0))
        acc_meter.update(acc.item(), x.size(0))
        if optimizer:
            loader_bar.set_description("Train epoch {}, loss: {:.4f}, acc: {:.4f}"
                                       .format(epoch, loss_meter.avg, acc_meter.avg))
        else:
            loader_bar.set_description("Test epoch {}, loss: {:.4f}, acc: {:.4f}"
                                       .format(epoch, loss_meter.avg, acc_meter.avg))

    return loss_meter.avg, acc_meter.avg


def get_lr(step, total_steps, lr_max, lr_min):
    print('step', step, 'total_steps', total_steps, 'lr_max', lr_max, 'lr_min', lr_min)
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


@hydra.main(version_base="1.3", config_path=None, config_name=None)
def finetune(args: DictConfig) -> None:
    print(OmegaConf.to_yaml(args))  # Print config for debugging
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Shared normalization values
    normalization_dict = {
        'CIFAR10': {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2470, 0.2435, 0.2616]},
        'CIFAR10C': {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2470, 0.2435, 0.2616]},
        'CIFAR100': {'mean': [0.5071, 0.4867, 0.4408], 'std': [0.2675, 0.2565, 0.2761]},
        'imagenet100': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
        'SVHN': {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]},
    }

    # Dataset-specific logic
    if args.dataset in ["CIFAR10", "CIFAR100"]:
        train_loader, test_loader, n_classes = load_cifar_datasets(args, normalization_dict)
        linear_evaluation_cifar(args, train_loader, test_loader, n_classes, device)
    elif args.dataset in ["imagenet100", "SVHN"]:
        train_loader, test_loader, n_classes = load_imagenet_svhn_datasets(args, normalization_dict)
        linear_evaluation_imagenet_svhn(args, train_loader, test_loader, n_classes, device)
    elif args.dataset == "CIFAR10C":
        train_loader, _, n_classes = load_cifar_datasets(args, normalization_dict)
        _, test_loader, n_classes = load_cifar10c_datasets(args, normalization_dict)
        linear_evaluation_cifar(args, train_loader, test_loader, n_classes, device)
        
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

def get_aug(args):
    train_transform_list = [transforms.RandomResizedCrop(32),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.ToTensor()]
    
    if args.strong_DA == "large_erase":
        # train_transform_list = [transforms.RandomResizedCrop(32),
        #                                 transforms.RandomApply([transforms.RandomResizedCrop(32, scale=(0.2, 0.2))], p=0.8),
        #                                   transforms.RandomHorizontalFlip(p=0.5),
        #                                   transforms.ToTensor()]
        test_transform_list = [transforms.ToTensor(),transforms.RandomErasing(scale=(0.10, 0.33), p=1)]
        train_transform_list.append(transforms.RandomErasing(scale=(0.10, 0.33), p=1))
        #test_transform_list = [transforms.ToTensor(),transforms.RandomErasing(scale=(0.10, 0.33), p=1)]
    elif args.strong_DA == "brightness":
        train_transform_list.append(Brightness(severity=5))  # Use the custom Brightness class
        train_transform_list.append(transforms.ToTensor())
        test_transform_list = [transforms.ToTensor(), Brightness(severity=5),transforms.ToTensor()]
    elif args.strong_DA == "strong_crop":
        train_transform_list.append(StrongCrop(img_size=32, severity=5)) 
        test_transform_list = [transforms.ToTensor(),StrongCrop(img_size=32, severity=5)]
        #test_transform_list = [transforms.ToTensor()]
    else:
        test_transform_list = [transforms.ToTensor()]
    return train_transform_list, test_transform_list

def load_cifar_datasets(args, normalization_dict, correputed_test=True):
    """Load CIFAR-10 or CIFAR-100 datasets with specific augmentations."""
    DatasetClass = torchvision.datasets.CIFAR100 if args.dataset == "CIFAR100" else torchvision.datasets.CIFAR10
    n_classes = 100 if args.dataset == "CIFAR100" else 10
    dataset_dir = hydra.utils.to_absolute_path(args.dataset_dir)
    
    if correputed_test is True:
        train_transform_list, test_transform_list = get_aug(args)
        train_transform_list.append(transforms.Normalize(normalization_dict[args.dataset]['mean'], normalization_dict[args.dataset]['std']))
        test_transform_list.append(transforms.Normalize(normalization_dict[args.dataset]['mean'], normalization_dict[args.dataset]['std']))
        train_transform = transforms.Compose(train_transform_list)
        test_transform = transforms.Compose(test_transform_list)
        print('Yeah, the train and test are corrupted')
    else:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(normalization_dict[args.dataset]['mean'], normalization_dict[args.dataset]['std']),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(normalization_dict[args.dataset]['mean'], normalization_dict[args.dataset]['std']),
        ])

    # Datasets
    train_dataset = DatasetClass(root=dataset_dir, train=True, transform=train_transform, download=True)
    test_dataset = DatasetClass(root=dataset_dir, train=False, transform=test_transform, download=True)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    return train_loader, test_loader, n_classes

def load_cifar10c_datasets(args, normalization_dict):
    root_dir = '../CIFAR-10-C'
    images, labels = load_and_merge_npy_files(root_dir, 'labels.npy')

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalization_dict[args.dataset]['mean'], normalization_dict[args.dataset]['std']),
    ])
    n_classes = 10

    # CIFAR-10C is typically used only as a test dataset
    testset = CIFAR10CDataset(images, labels, transform=test_transform)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    # Return None for train_loader since we're not training on CIFAR-10C
    return None, test_loader, n_classes

def load_imagenet_svhn_datasets(args, normalization_dict):
    """Load ImageNet-100 or SVHN datasets with specific augmentations."""
    dataset_dir = hydra.utils.to_absolute_path(args.dataset_dir)
    if args.dataset == "imagenet100":
        
        train_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            #transforms.Normalize(normalization_dict['imagenet100']['mean'], normalization_dict['imagenet100']['std']),
        ])
        train_dataset = ImageFolder(root=f"{dataset_dir}/train", transform=train_transform)
        test_dataset = ImageFolder(root=f"{dataset_dir}/val", transform=train_transform)
        n_classes = 100
    elif args.dataset == "SVHN":
        train_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(normalization_dict['SVHN']['mean'], normalization_dict['SVHN']['std']),
        ])
        train_dataset = torchvision.datasets.SVHN(root=dataset_dir, split="train", transform=train_transform, download=True)
        test_dataset = torchvision.datasets.SVHN(root=dataset_dir, split="test", transform=train_transform, download=True)
        n_classes = 10

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.logistic_batch_size, shuffle=True, drop_last=True, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.logistic_batch_size, shuffle=False, drop_last=False, num_workers=args.workers)

    return train_loader, test_loader, n_classes


def linear_evaluation_cifar(args, train_loader, test_loader, n_classes, device):
    """Linear evaluation function for CIFAR-10 and CIFAR-100."""
    # Shared evaluation logic for CIFAR datasets

    cl_model = ContrastiveLearning(args)
    if args.pretrained_model_path is not None:
        #cl_model.load_state_dict(torch.load(args.pretrained_model_path)["state_dict"],strict=False)
        cl_model.load_state_dict(torch.load(args.pretrained_model_path, weights_only=False),strict=False)
    else:
        if args.dataset == "CIFAR10C":
            cl_model.load_state_dict(torch.load(f"model/{args.task}_{args.backbone}_da_{args.strong_DA}_seed{args.seed}_epoch={args.load_epoch-1}_CIFAR10.ckpt", weights_only=False)["state_dict"],strict=True)
        else:
            cl_model.load_state_dict(torch.load(f"model/{args.task}_{args.backbone}_da_{args.strong_DA}_seed{args.seed}_epoch={args.load_epoch-1}_{args.dataset}.ckpt", weights_only=False)["state_dict"],strict=True)
    pre_model=cl_model.model
    model = LinModel(pre_model.enc, feature_dim=pre_model.feature_dim, n_classes=n_classes).to(device)
    model.enc.requires_grad = False

    # Optimizer and Scheduler
    parameters = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=0.2, momentum=args.momentum, weight_decay=0.0, nesterov=True)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: get_lr(step, args.finetune_epochs * len(train_loader), args.learning_rate, 1e-3))

    # Evaluation Loop
    evaluate_model(args, model, train_loader, test_loader, optimizer, scheduler, device)


def linear_evaluation_imagenet_svhn(args, train_loader, test_loader, n_classes, device):
    """Linear evaluation function for ImageNet-100 and SVHN."""
    # Shared evaluation logic for finetune_epochs/SVHN datasets
    encoder = get_resnet(args.backbone, pretrained=False)
    n_features = encoder.fc.in_features
    cl_model = ContrastiveLearning(args)
    pre_model = cl_model.model.to(device)
    if args.pretrained_model_path is not None:
        print('loading from pretrained model',args.pretrained_model_path)
        checkpoint = torch.load(args.pretrained_model_path, weights_only=False)
        state_dict = checkpoint['state_dict']
        #cl_model.load_state_dict(torch.load(args.pretrained_model_path)["state_dict"],strict=False)
    else:
        checkpoint = torch.load(f"model/{args.task}_{args.backbone}_da_{args.strong_DA}_seed{args.seed}_epoch={args.load_epoch-1}_{args.dataset}.ckpt", weights_only=False)
        state_dict = checkpoint['state_dict']
        #cl_model.load_state_dict(torch.load(f"model/{args.task}_{args.backbone}_da_{args.strong_DA}_seed{args.seed}_epoch={args.load_epoch-1}_{args.dataset}.ckpt")["state_dict"],strict=True)
    for key in list(state_dict.keys()):
        state_dict[key.replace('model.enc', 'model.encoder')] = state_dict.pop(key)
    try:
        cl_model.load_state_dict(state_dict, strict=False)
        print("Pre-trained SimCLR model loaded successfully.")
    except KeyError as e:
        print(f"KeyError: {e}")
        print("Attempting to load with strict=False...")
        pre_model.load_state_dict(state_dict, strict=False)
        print("Pre-trained SimCLR model loaded with some missing or unexpected keys.")
    pre_model = cl_model.model.to(device)
    pre_model.eval()

    # Logistic Regression Model
    model = nn.Linear(n_features, n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()
    print("### Creating features from pre-trained context model ###")

    # Feature Extraction
    (train_X, train_y, test_X, test_y) = get_features(pre_model, train_loader, test_loader, device)
    arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(train_X, train_y, test_X, test_y, args.logistic_batch_size)

    # Logistic Regression Training
    train_logistic_regression(args, arr_train_loader, arr_test_loader, pre_model, model, criterion, optimizer, device)


def evaluate_model(args, model, train_loader, test_loader, optimizer, scheduler, device):
    """Shared evaluation loop for CIFAR datasets."""
    optimal_loss, optimal_acc = 1e5, 0.0
    for epoch in range(args.finetune_epochs):
        train_loss, train_acc = run_epoch(model, train_loader, epoch, optimizer, scheduler)
        test_loss, test_acc = run_epoch(model, test_loader, epoch)

        if test_acc > optimal_acc:
            optimal_loss = train_loss
            optimal_acc = test_acc
            logger.info("==> New best results")
            torch.save(model.state_dict(), f"model/{args.task}_lin_{args.backbone}_{args.strong_DA}_best_seed{args.seed}_{args.dataset}.pth")

    logger.info(f"Best Test Acc: {optimal_acc:.4f}")


def train_logistic_regression(args, train_loader, test_loader,simclr_model ,model, criterion, optimizer, device):
    """Train and evaluate logistic regression for ImageNet-100/SVHN."""
    for epoch in range(args.finetune_epochs):
        loss_epoch, acc_epoch = train(device, train_loader, simclr_model, model, criterion, optimizer)
        print(
            f"Epoch [{epoch}/{args.finetune_epochs}]\t Loss: {loss_epoch / len(train_loader)}\t Accuracy: {acc_epoch / len(train_loader)}"
        )

    # Final testing
    loss_epoch, acc_epoch = test(device, test_loader, simclr_model, model, criterion, optimizer)
    print(
        f"[FINAL]\t Loss: {loss_epoch / len(test_loader)}\t Accuracy: {acc_epoch / len(test_loader)}"
    )

  







def inference(loader, simclr_model, device):
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            h, z = simclr_model(x)

        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


def get_features(simclr_model, train_loader, test_loader, device):
    train_X, train_y = inference(train_loader, simclr_model, device)
    test_X, test_y = inference(test_loader, simclr_model, device)
    return train_X, train_y, test_X, test_y


def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False
    )

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def train(device, loader, simclr_model, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        # if step % 100 == 0:
        #     print(
        #         f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t Accuracy: {acc}"
        #     )

    return loss_epoch, accuracy_epoch


def test(device, loader, simclr_model, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    model.eval()
    for step, (x, y) in enumerate(loader):
        model.zero_grad()

        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss_epoch += loss.item()

    return loss_epoch, accuracy_epoch


if __name__ == "__main__":
    finetune()