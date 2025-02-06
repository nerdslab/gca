# Generalized Contrastive Alignment (GCA) - NeurIPS 2024

Generalized Contrastive Alignment (GCA) provides a robust framework for self-supervised learning tasks, supporting various datasets and augmentation methods.

---

## **Environment Setup**
To set up the required environment, follow these steps:

```bash
# Create and activate the environment
conda create -n GCA python=3.11.9 -y
conda activate GCA

# Install dependencies
pip install hydra-core numpy==1.26.4 matplotlib seaborn scikit-image scikit-learn \
            pytorch-lightning==1.9.5 torch==2.2.1 torchaudio==2.2.1 \
            torchmetrics==1.4.2 torchvision==0.17.1
```

---

## **References**
- **SimCLR CIFAR10 Implementation** by Damrich et al.: [GitHub Link](https://github.com/p3i0t/SimCLR-CIFAR10)
- **SimCLR** by Ting Chen et al.: [GitHub Link](https://github.com/Spijkervet/SimCLR)
- **IOT** in Liangliang Shi, et al. "Understanding and generalizing contrastive learning from the inverse optimal transport perspective." ICML, 2023.
---

## **Usage Instructions**

### **Supported Tasks**
The framework supports the following tasks:
- `simclr`
- `hs_ince`
- `gca_ince`
- `rince`
- `gca_rince`
- `gca_uot`

### **Supported Datasets**
The following datasets are supported:
- `SVHN`
- `imagenet100`
- `cifar100`
- `cifar10`

### **Strong Data Augmentation**
You can configure data augmentation using the `strong_DA` option:
- `None` (standard augmentation)
- `large_erase`
- `brightness`
- `strong_crop`

---

## **Pretraining a Model**
To pretrain a model using self-supervised learning, run the following script:

```bash
python ssl_pretrain.py \
    --config-name "simclr_cifar10.yaml" \
    --config-path "./config/" \
    task=gca_uot \
    dataset=CIFAR10 \
    dataset_dir="./datasets" \
    batch_size=512 \
    seed=32 \
    backbone=resnet18 \
    projection_dim=128 \
    strong_DA=None \
    gpus=1 \
    workers=16 \
    optimizer='SGD' \
    learning_rate=0.03 \
    momentum=0.9 \
    weight_decay=1e-6 \
    lam=0.01 \
    q=0.6 \
    max_epochs=500 \
    r1=1 \
    r2=0.2
```

---

## **Linear Evaluation**
To evaluate the pretrained model with a linear classifier, use the following script:

```bash
python linear_evaluation.py \
    --config-name="simclr_cifar10.yaml" \
    --config-path="./config/" \
    task=gca_uot \
    dataset=cifar10 \
    batch_size=512 \
    seed=64 \
    backbone=resnet18 \
    projection_dim=128 \
    strong_DA=None \
    lam=0.01 \
    q=0.6 \
    load_epoch=500
```

---

## **Configuration Options**
- **Task:** Specify the self-supervised learning task (e.g., `gca_uot`).
- **Dataset:** Choose from supported datasets (e.g., `cifar10`).
- **Data Augmentation:** Use `strong_DA` to set augmentation type.
- **Training Parameters:**
  - `batch_size`: Batch size for training.
  - `backbone`: Backbone architecture (e.g., `resnet18`).
  - `projection_dim`: Dimension of the projection head.
  - `lam` and `q`: Regularization and scaling parameters.
  - `max_epochs`: Maximum number of epochs for training.

---

## **Notes**
- Ensure that the `dataset_dir` contains the datasets in the correct structure.
- Customize parameters in the scripts to fit your experimental needs. As an example for SVHN,


```bash
python ssl_pretrain.py \
    --config-name "simclr_svhn.yaml" \
    --config-path "./config/" \
    task=gca_uot \
    dataset=SVHN \
    dataset_dir="./datasets" \
    batch_size=512 \
    seed=48 \
    backbone=resnet18 \
    projection_dim=128 \
    strong_DA=None \
    gpus=1 \
    workers=16 \
    optimizer='Adam' \
    learning_rate=0.03 \
    momentum=0.9 \
    weight_decay=1e-6 \
    lam=0.01 \
    q=0.6 \
    max_epochs=500 \
    r1=1 \
    r2=0.01
```

## **Citation**

If you find this repository helpful for your research, **please cite** our paper:

```bibtex
@inproceedings{chenyour,
  title={Your contrastive learning problem is secretly a distribution alignment problem},
  author={Chen, Zihao and Lin, Chi-Heng and Liu, Ran and Xiao, Jingyun and Dyer, Eva L},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems}
}