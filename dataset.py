import torch
import numpy as np

from torchvision import datasets, transforms

# class Config:
#     imagenette = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]

#     # ["australian_terrier", "border_terrier", "samoyed", "beagle", "shih-tzu", "english_foxhound", "rhodesian_ridgeback", "dingo", "golden_retriever", "english_sheepdog"]
#     imagewoof = [193, 182, 258, 162, 155, 167, 159, 273, 207, 229]

#     # ["tabby_cat", "bengal_cat", "persian_cat", "siamese_cat", "egyptian_cat", "lion", "tiger", "jaguar", "snow_leopard", "lynx"]
#     imagemeow = [281, 282, 283, 284, 285, 291, 292, 290, 289, 287]

#     # ["peacock", "flamingo", "macaw", "pelican", "king_penguin", "bald_eagle", "toucan", "ostrich", "black_swan", "cockatoo"]
#     imagesquawk = [84, 130, 88, 144, 145, 22, 96, 9, 100, 89]

#     # ["pineapple", "banana", "strawberry", "orange", "lemon", "pomegranate", "fig", "bell_pepper", "cucumber", "green_apple"]
#     imagefruit = [953, 954, 949, 950, 951, 957, 952, 945, 943, 948]

#     # ["bee", "ladys slipper", "banana", "lemon", "corn", "school_bus", "honeycomb", "lion", "garden_spider", "goldfinch"]
#     imageyellow = [309, 986, 954, 951, 987, 779, 599, 291, 72, 11]

#     dict = {
#         "imagenette" : imagenette,
#         "imagewoof" : imagewoof,
#         "imagefruit": imagefruit,
#         "imageyellow": imageyellow,
#         "imagemeow": imagemeow,
#         "imagesquawk": imagesquawk,
#     }

# config = Config()

def get_dataset(dataset, data_path, batch_size, subset=None, args=None):
    class_map = None
    loader_train_dict = None
    class_map_inv = None

    if dataset == 'MNIST':
        channel = 1
        im_size = 28
        num_classes = 10
        mean = [0.1307]
        std = [0.3081]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
        dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes
        class_map = {x:x for x in range(num_classes)}

    elif dataset == 'CIFAR10':
        channel = 3
        im_size = 32
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2024, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
        dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes
        class_map = {x:x for x in range(num_classes)}

        # filter for CIFAR10 subset
        if subset:
            # find the index of the class to filter by 
            subset_class_index = class_names.index(subset)
            # filter the dataset to only include images of the specified class
            dst_train = [sample for sample in dst_train if sample[1] == subset_class_index]
            dst_test = [sample for sample in dst_test if sample[1] == subset_class_index]
            # convert filtered list back to a Dataset
            dst_train = torch.utils.data.Subset(dst_train, indices=range(len(dst_train)))
            dst_test = torch.utils.data.Subset(dst_test, indices=range(len(dst_test)))


    # elif dataset == 'ImageNet':
    #     channel = 3
    #     im_size = 128
    #     num_classes = 10

    #     config.img_net_classes = config.dict[imagenet_subset]

    #     mean = [0.485, 0.456, 0.406]
    #     std = [0.229, 0.224, 0.225]
    #     transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=mean, std=std),
    #         transforms.Resize((im_size, im_size)),
    #         transforms.CenterCrop((im_size, im_size))
    #     ])
    #     dst_train = datasets.ImageNet(data_path, split="train", transform=transform) # no augmentation
    #     dst_train_dict = {c : torch.utils.data.Subset(dst_train, np.squeeze(np.argwhere(np.equal(dst_train.targets, config.img_net_classes[c])))) for c in range(len(config.img_net_classes))}
    #     dst_train = torch.utils.data.Subset(dst_train, np.squeeze(np.argwhere(np.isin(dst_train.targets, config.img_net_classes))))
    #     loader_train_dict = {c : torch.utils.data.DataLoader(dst_train_dict[c], batch_size=batch_size, shuffle=True, num_workers=16) for c in range(len(config.img_net_classes))}
    #     dst_test = datasets.ImageNet(data_path, split="val", transform=transform)
    #     dst_test = torch.utils.data.Subset(dst_test, np.squeeze(np.argwhere(np.isin(dst_test.targets, config.img_net_classes))))
    #     for c in range(len(config.img_net_classes)):
    #         dst_test.dataset.targets[dst_test.dataset.targets == config.img_net_classes[c]] = c
    #         dst_train.dataset.targets[dst_train.dataset.targets == config.img_net_classes[c]] = c
    #     print(dst_test.dataset)
    #     class_map = {x: i for i, x in enumerate(config.img_net_classes)}
    #     class_map_inv = {i: x for i, x in enumerate(config.img_net_classes)}
    #     class_names = None


    testloader = torch.utils.data.DataLoader(dst_test, batch_size=batch_size, shuffle=False, num_workers=2)

    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv
