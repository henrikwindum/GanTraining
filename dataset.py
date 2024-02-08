import torch
import numpy as np

from torchvision import datasets, transforms

def get_dataset(args):
    if args.dataset == 'MNIST':
        channel = 1
        im_size = 28
        num_classes = 10
        mean = [0.1307]
        std = [0.3081]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.MNIST(args.path, train=True, download=True, transform=transform)
        dst_test = datasets.MNIST(args.path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif args.dataset == 'CIFAR10':
        channel = 3
        im_size = 32
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2024, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR10(args.path, train=True, download=True, transform=transform)
        dst_test = datasets.CIFAR10(args.path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

        # filter for CIFAR10 subset
        if args.subset:
            # find the index of the class to filter by 
            subset_class_index = class_names.index(args.subset)
            # filter the dataset to only include images of the specified class
            dst_train = [sample for sample in dst_train if sample[1] == subset_class_index]
            dst_test = [sample for sample in dst_test if sample[1] == subset_class_index]
            # convert filtered list back to a Dataset
            dst_train = torch.utils.data.Subset(dst_train, indices=range(len(dst_train)))
            dst_test = torch.utils.data.Subset(dst_test, indices=range(len(dst_test)))

    testloader = torch.utils.data.DataLoader(dst_test, batch_size=args.batch_real, shuffle=False, num_workers=args.workers)

    return channel, im_size, num_classes, class_names, mean, std, dst_train, testloader
