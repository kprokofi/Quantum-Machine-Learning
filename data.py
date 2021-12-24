import torch
from torchvision import datasets, transforms


def get_train_val_datasets(root_path):
    transform_dict = {
        'train': transforms.Compose(
        [transforms.Resize((224,224)),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
         ]),
        'test': transforms.Compose(
        [transforms.Resize((224,224)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
         ])}
    val_dataset = datasets.ImageFolder(root=root_path + '/train', transform=transform_dict['train'])
    train_dataset = datasets.ImageFolder(root=root_path + '/test', transform=transform_dict['test'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=False, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, drop_last=False, num_workers=4)

    # n_samples = 100

    # X_train = datasets.MNIST(root='./data', train=True, download=True,
    #                         transform=transforms.Compose([transforms.ToTensor()]))

    # idx = np.append(np.where(X_train.targets == 0)[0][:n_samples],
    #                 np.where(X_train.targets == 8)[0][:n_samples])

    # X_train.data = X_train.data[idx]
    # X_train.targets = X_train.targets[idx]

    # train_loader = torch.utils.data.DataLoader(X_train, batch_size=1, shuffle=True)

    # n_samples = 50

    # X_test = datasets.MNIST(root='./data', train=False, download=True,
    #                         transform=transforms.Compose([transforms.ToTensor()]))

    # idx = np.append(np.where(X_test.targets == 0)[0][:n_samples],
    #                 np.where(X_test.targets == 8)[0][:n_samples])

    # X_test.data = X_test.data[idx]
    # X_test.targets = X_test.targets[idx]

    # test_loader = torch.utils.data.DataLoader(X_test, batch_size=1, shuffle=True)

    return train_loader, val_loader