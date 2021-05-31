import torch
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import os

# calculate Normalize
def cal_nor(data_loader):
    nb_samples = 0.
    channel_mean = torch.zeros(3)
    channel_std = torch.zeros(3)

    for images, targets in data_loader:
        # scale image to be between 0 and 1
        N, C, H, W = images.shape[:4]
        data = images.view(N, C, -1)

        channel_mean += data.mean(2).sum(0)
        channel_std += data.std(2).sum(0)
        nb_samples += N

    channel_mean /= nb_samples
    channel_std /= nb_samples
    return channel_mean, channel_std


def creat_loader(train_data_dir, test_data_dir, batch_size, resize):
    assert os.path.exists(train_data_dir), f'{train_data_dir} not exist!'
    train_img_data = ImageFolder(train_data_dir,
                                                    transform=transforms.Compose([
                                                        transforms.Resize((resize,resize)), 
                                                        transforms.ToTensor()
                                                        ])
                                                    )
    data_loader = torch.utils.data.DataLoader(train_img_data, batch_size=batch_size, shuffle=False)
    
    channel_mean, channel_std = cal_nor(data_loader)
    
    # creat dataloader
    # trainset
    assert os.path.exists(train_data_dir), f'{train_data_dir} not exist!'
    train_img_data = ImageFolder(train_data_dir,
                                                transform=transforms.Compose([
                                                    transforms.Resize((resize,resize)), 
                                                    transforms.RandomRotation(10),
                                                    transforms.ToTensor(),     
                                                    transforms.Normalize(channel_mean, channel_std)
                                                    ])
                                                )
    # train loader
    train_data_loader = torch.utils.data.DataLoader(train_img_data, batch_size=batch_size, shuffle=True)
    

    # testset
    assert os.path.exists(test_data_dir), f'{test_data_dir} not exist!'
    test_img_data = ImageFolder(test_data_dir,
                                                transform=transforms.Compose([
                                                    transforms.Resize((resize,resize)),
                                                    transforms.RandomRotation(10), 
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(channel_mean, channel_std)
                                                    ])
                                                )
    # test loader
    test_data_loader = torch.utils.data.DataLoader(test_img_data, batch_size=batch_size, shuffle=False)
    return train_img_data, train_data_loader, test_img_data, test_data_loader