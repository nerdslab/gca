

import os
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




def load_and_merge_npy_files(root_dir, label_file):
    image_files = [f for f in os.listdir(root_dir) if f.endswith('.npy') and f != label_file]
    labels = np.load(os.path.join(root_dir, label_file))
    all_images = []
    repeated_labels=[]
    for img_file in image_files:
        if img_file.startswith('snow'):
            images = np.load(os.path.join(root_dir, img_file))
            #print('images.shape',images.shape)
            all_images.append(images)
            repeated_labels.append(labels)
    
    return np.concatenate(all_images), np.concatenate(repeated_labels),
