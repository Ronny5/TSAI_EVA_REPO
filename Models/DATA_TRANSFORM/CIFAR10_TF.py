def cifar10_mean_std():
    """Return the true mean of entire test and train dataset"""
    # simple transform
    simple_transforms = transforms.Compose([transforms.ToTensor(),])
    
    exp_train = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=simple_transforms)
    exp_test = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=simple_transforms)
    
    exp_tr_data = exp_train.data # train set
    exp_ts_data = exp_test.data # test set
    
    exp_data = np.concatenate((exp_tr_data,exp_ts_data),axis=0) # contatenate entire data
    
    exp_data = np.transpose(exp_data,(3,1,2,0)) # reshape to (60000, 32, 32, 3)
    
    norm_mean = (np.mean(exp_data[0])/255, np.mean(exp_data[1])/255, np.mean(exp_data[2])/255)
    norm_std   = (np.std(exp_data[0])/255, np.std(exp_data[1])/255, np.std(exp_data[2])/255)
    
    return(tuple(map(lambda x: np.round(x,2), norm_mean)), tuple(map(lambda x: np.round(x,2), norm_std)))
    


# The output of torchvision datasets are PILImage images of range [0, 1]. We transform them to Tensors of normalized range [-1, 1].
# Define Transforms
norm_mean,norm_std = cifar10_mean_std() #  (0.49, 0.48, 0.45), (0.25, 0.24, 0.26)
train_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(norm_mean, norm_std)])
test_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(norm_mean, norm_std)])
