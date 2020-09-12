




#########################################
#      Session 7 Assignment Utils       #
#########################################

import torch
import torchvision
import torchvision.transforms as transforms

def cifar10_mean_std(dsname):
    """Return the true mean of entire test and train dataset"""
    # simple transform
    simple_transforms = transforms.Compose([transforms.ToTensor(),])
    
    exp_train = torchvision.datasets.dsname('./data', train=True, download=True, transform=simple_transforms)
    exp_test = torchvision.datasets.dsname('./data', train=False, download=True, transform=simple_transforms)
    
    exp_tr_data = exp_train.data # train set
    exp_ts_data = exp_test.data # test set
    
    exp_data = np.concatenate((exp_tr_data,exp_ts_data),axis=0) # contatenate entire data
    
    exp_data = np.transpose(exp_data,(3,1,2,0)) # reshape to (60000, 32, 32, 3)
    
    norm_mean = (np.mean(exp_data[0])/255, np.mean(exp_data[1])/255, np.mean(exp_data[2])/255)
    norm_std   = (np.std(exp_data[0])/255, np.std(exp_data[1])/255, np.std(exp_data[2])/255)
    
    return(tuple(map(lambda x: np.round(x,2), norm_mean)), tuple(map(lambda x: np.round(x,2), norm_std)))



def train(norm_mean=0.5, norm_std=0.5, rotx = -5.0, roty = 5.0, hflip = 0.5):
    ''' Parameter Sequence : norm_mean=0.5, norm_std=0.5, rotx = -5.0, roty = 5.0, hflip = 0.5'''
    train_transform = transforms.Compose([transforms.RandomRotation((rotx, roty), fill=(1,1,1)),
                                      transforms.RandomHorizontalFlip(p=hflip),
                                      transforms.ToTensor(),
                                      transforms.Normalize(norm_mean, norm_std)])
    return train_transform

def test():
    test_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(norm_mean, norm_std)])
    return test_transform


