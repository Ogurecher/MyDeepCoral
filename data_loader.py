from torchvision import datasets, transforms
import torch
import settings

def load_training(root_path, dir, batch_size):
    transform = transforms.Compose(
        [transforms.Resize([64, 64]),
         transforms.Grayscale(num_output_channels=1),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader

def load_testing(root_path, dir, batch_size):
    transform = transforms.Compose(
        [transforms.Resize([64, 64]),
         transforms.Grayscale(num_output_channels=1),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    return test_loader


source_loader = load_training(settings.root_path, settings.source_name, settings.batch_size_train)
target_train_loader = load_training(settings.root_path, settings.target_name, settings.batch_size_train)
target_test_loader = load_testing(settings.root_path, settings.target_name, settings.batch_size_test)

len_source_dataset = len(source_loader.dataset)
len_target_dataset = len(target_test_loader.dataset)
len_source_loader = len(source_loader)
len_target_loader = len(target_train_loader)
