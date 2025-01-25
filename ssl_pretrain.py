#import logging
from PIL import Image

import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import torch.backends.cudnn as cudnn

from pytorch_lightning import Trainer, LightningModule

from cl_models.simclr import SimCLR, Brightness, StrongCrop
from cl_models.modules import get_resnet
from cl_models.modules.transformations import TransformsSimCLR

import hydra
import importlib
from omegaconf import DictConfig,OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
#logger = logging.getLogger(__name__)




class CIFAR100Pair(CIFAR100):
    """Generate mini-batch pairs on CIFAR100 training set."""
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)  # Convert the numpy array image to PIL image
        imgs = [self.transform(img), self.transform(img)]  # Apply the same transform twice to get a pair
        return torch.stack(imgs), target  # Stack the pair of images and return with the target


class CIFAR10Pair(CIFAR10):
    """Generate mini-batche pairs on CIFAR10 training set."""
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)  # .convert('RGB')
        imgs = [self.transform(img), self.transform(img)]
        return torch.stack(imgs), target  # stack a positive pair




def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


# color distortion composed by color jittering and color dropping.
# See Section A of SimCLR: https://arxiv.org/abs/2002.05709
def get_color_distortion(s=0.5):  # 0.5 for CIFAR10 by default
    # s is the strength of color distortion
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort






class ContrastiveLearning(LightningModule):
    def __init__(self, args,loss_function=None):
        super().__init__()
        # self.hparams = args
        self.save_hyperparameters(args)
        # initialize ResNet
        self.encoder = get_resnet(self.hparams.backbone, pretrained=False)
        if args.backbone == 'resnet18':
            modified_conv1 = True
        else:
            modified_conv1 = False
        self.model = SimCLR(self.encoder, self.hparams.projection_dim, modified_conv1)
        # Select the loss function based on the task
        self.loss_function = loss_function#globals()[args.task](args.batch_size, args.temperature,  args.iterations,args.lam, args.q)


    def training_step(self, batch, batch_idx):
        # Check the structure of the batch
        if isinstance(batch[0], list):  # Handle case for (x_i, x_j), _
            (x_i, x_j), _ = batch
            x = torch.cat((x_i, x_j), dim=0)
        else:  # Handle case for x, _
            x, _ = batch
            sizes = x.size()
            x = x.view(sizes[0] * 2, sizes[2], sizes[3], sizes[4])

        h, z = self.model(x)
        loss = self.loss_function(z)
        return loss

    def configure_optimizers(self):
        scheduler = None
        if self.hparams.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        elif self.hparams.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                        self.model.parameters(),
                        0.2,
                        momentum=self.hparams.momentum,
                        weight_decay=self.hparams.weight_decay,
                        nesterov=True)
                    # cosine annealing lr
            scheduler = LambdaLR(
                optimizer,
                lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
                    step,
                    self.hparams.max_epochs * self.hparams.len_train_loader,
                    self.hparams.learning_rate,  # lr_lambda computes multiplicative factor
                    1e-3))
        if scheduler:
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return {"optimizer": optimizer}

task_function_mapping = {
    "iot": "iot",
    "iot_uni": "iot_uni",
    "hs_ince": "hs_ince",
    "gca_ince": "gca_ince",
    "gca_uot": "gca_uot",
    "rince": "rince",
    "hs_rince": "hs_rince",
    "gca_rince": "gca_rince",
    "simclr": "NT_Xent",
}



def get_dataset(args):
    dataset_dir = hydra.utils.to_absolute_path(args.dataset_dir)  # get absolute path of data dirs

    if args.dataset == 'CIFAR10':
        train_transform_list = get_aug(args)
        train_transform_list.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        train_transform = transforms.Compose(train_transform_list)
        train_dataset = CIFAR10Pair(root=dataset_dir,
                                train=True,
                                transform=train_transform,
                                download=True)
    elif args.dataset == 'CIFAR100':
        train_transform_list = get_aug(args)
        train_transform_list.append(transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)))
        train_transform = transforms.Compose(train_transform_list)
        train_dataset = CIFAR100Pair(root=dataset_dir,
                                train=True,
                                transform=train_transform,
                                download=True)
    elif args.dataset == "STL10":
        train_dataset = torchvision.datasets.STL10(
            root=dataset_dir,
            split="unlabeled",
            download=True,
           transform=TransformsSimCLR(size=args.image_size),
        )
    elif args.dataset == "imagenet100":
        train_dataset = ImageFolder(root=f'{dataset_dir}/train',  transform=TransformsSimCLR(size=args.image_size))
    elif args.dataset == "SVHN":
        train_dataset = torchvision.datasets.SVHN(
            root=dataset_dir,
            split="train",
            download=True,
            transform=TransformsSimCLR(size=args.image_size),
        )
    return train_dataset

def get_aug(args):
    train_transform_list = [
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        get_color_distortion(s=0.5),
        transforms.ToTensor()
    ]

    if args.strong_DA == "large_erase":
        #train_transform_list.append(transforms.RandomErasing(scale=(0.10, 0.33), p=1))        
        train_transform_list = [
        transforms.RandomResizedCrop(32),
        transforms.RandomApply([transforms.RandomResizedCrop(32, scale=(0.2, 0.2))], p=0.8),
        transforms.RandomHorizontalFlip(p=0.5),
        get_color_distortion(s=0.5),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor() ]
    elif args.strong_DA == "brightness":
        train_transform_list.append(Brightness(severity=5))  # Use the custom Brightness class
        train_transform_list.append(transforms.ToTensor())
    elif args.strong_DA == "strong_crop":
        train_transform_list.append(StrongCrop(img_size=32, severity=5))  # Use the custom StrongCrop class
    print(f'The strong augmentation here is{args.strong_DA}')
    return train_transform_list





@hydra.main(version_base="1.3", config_path=None, config_name=None)
def ssl_pretrain(args: DictConfig) -> None:
    #print(OmegaConf.to_yaml(args))  # Optional: Print the loaded configuration for debugging
    assert torch.cuda.is_available(), "CUDA is not available."
    torch.manual_seed(args.seed)
    cudnn.benchmark = True
    print('yes')
    # Import the appropriate module based on the dataset
    if args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100':
        module_name = 'gca_loss_cifar'
    elif args.dataset == 'imagenet100' or args.dataset == 'SVHN':
        module_name = 'gca_loss_imagenet'
    else:
        raise ValueError(f"Dataset {args.dataset} is not supported.")
    # Dynamically import the required module
    try:
        loss_module = importlib.import_module(module_name)
        selected_function_name = task_function_mapping[args.task]
        selected_function = getattr(loss_module, selected_function_name)
    except AttributeError as e:
       raise AttributeError(f"The function '{selected_function_name}' is missing in the module '{module_name}'.") from e
    if args.task == 'gca_uot':
        loss_function=selected_function(args.batch_size, args.epsilon,  args.iterations,args.lam, args.q,args.relax_item1,args.relax_item2, args.r1,args.r2)
    else:
        loss_function=selected_function(args.batch_size, args.epsilon,  args.iterations,args.lam, args.q)
    train_dataset = get_dataset(args)

    args.accelerator  = "gpu"
    args.strategy = "ddp" 
    
    workers = args.workers
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=workers, drop_last=True)    
    args.len_train_loader = len(train_loader)
    args.epsilon = 0.5
    args.iterations = 10
    cl = ContrastiveLearning(args,loss_function)
    
    checkpoint_callback = ModelCheckpoint(
    dirpath='model/',      # Directory where checkpoints are saved
    filename= '{}_{}_da_{}_seed{}_{{epoch}}_{}'.format(args.task,args.backbone,args.strong_DA,args.seed,args.dataset),  # Checkpoint filename
    save_top_k=-1,               # Save all checkpoints
    every_n_epochs=20,           # Save every 100 epochs
    save_on_train_epoch_end=True
    )

    trainer = Trainer(
        accelerator= args.accelerator,
        devices=args.gpus,
        strategy=args.strategy,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback],
        sync_batchnorm=True,
        logger = False,
    )
    trainer.fit(cl, train_loader)
    
if __name__ == "__main__":
    ssl_pretrain()


