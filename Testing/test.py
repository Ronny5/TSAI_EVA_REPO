







#########################################
#      		Test Module		 	        #
#########################################


def test(dsname,transform_var):
	test_set  = torchvision.datasets.dsname(root='./data', train=False,download=True, transform=transform_var)
	return test_set


def s7testdataloader(test_set, dataloader_args):
	# test dataloader
	test_loader  = torch.utils.data.DataLoader(test_set, **dataloader_args)
	return test_loader

	