







#########################################
#      		Test Module		 	        #
#########################################

import torch
import torchvision
import torchvision.transforms as transforms

def test(transform_var):
	test_set  = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform_var)
	return test_set


def s7dataloader(test_set, **dataloader_args):
	# test dataloader
	test_loader  = torch.utils.data.DataLoader(test_set, **dataloader_args)
	return test_loader

	
