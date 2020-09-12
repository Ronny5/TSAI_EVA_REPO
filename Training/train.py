





#########################################
#      		Train Module		        #
#########################################

import torch
import torchvision
import torchvision.transforms as transforms

# Assignment 7

def s7train(transform_var):
	train_set = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform_var)
	return train_set


def s7dataloader(train_set, dataloader_args):
    # train dataloader
    train_loader = torch.utils.data.DataLoader(train_set, **dataloader_args)
    return


