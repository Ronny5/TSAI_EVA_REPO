





#########################################
#      Session 7 Assignment Misc        #
#########################################

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def samplegrid(trainloader):
	# get some random training images
	dataiter = iter(trainloader)
	images, labels = dataiter.next()

	# show images
	imshow(torchvision.utils.make_grid(images))
	# print labels
	print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
