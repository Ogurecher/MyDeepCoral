import numpy as np
from torch.utils import data
from torchvision import datasets, transforms

import utils


# Spilt a dataset into training and testing dataset.
# Returns: train_loader, test_loader
def get_train_test_loader(directory, batch_size, testing_size=0.1, img_size=None):
    mean, std = utils.get_dataset_mean_and_std(directory)
    #print(mean, std)
    transform = [
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]

    if img_size is not None:
        transform.insert(0, transforms.Resize(img_size))

    dataset = datasets.ImageFolder(
        directory,
        transform=transforms.Compose(transform)
    )

    num_data = len(dataset)
    indices = list(range(num_data))
    split = int(np.floor(testing_size * num_data))

    # Shuffle
    np.random.shuffle(indices)

    train_idx, test_idx = indices[split:], indices[:split]

    train_sampler = data.sampler.SubsetRandomSampler(train_idx)
    test_sampler = data.sampler.SubsetRandomSampler(test_idx)

    train_loader = data.DataLoader(
        dataset,
        num_workers=2,
        batch_size=batch_size,
        sampler=train_sampler,
        drop_last=True
    )

    test_loader = data.DataLoader(
        dataset,
        num_workers=2,
        batch_size=batch_size,
        sampler=test_sampler,
        drop_last=True
    )

    return train_loader, test_loader


# For Office31 datasets data_loader
def get_office31_dataloader(case, batch_size):
    print('[INFO] Loading datasets: {}'.format(case))
    datas = {
        'source': '../LungCancerdetection/src/data/train',
        #'dslr': 'dataset/office31/dslr/images/',
        'target': '../LungCancerdetection/src/data/test'
    }
    means = {
        'source': [0.27116661178406126, 0.27116661178406126, 0.27116661178406126],
        'target': [0.27470280839596517, 0.27470280839596517, 0.27470280839596517],
        #'dslr': [],
        'imagenet': [0.485, 0.456, 0.406] #[0.5, 0.5, 0.5]?
    }
    stds = {
        'source': [0.22253648754736358, 0.22253648754736358, 0.22253648754736358],
        'target': [0.2197482351411212, 0.2197482351411212, 0.2197482351411212],
        #'dslr': [],
        'imagenet': [0.229, 0.224, 0.225]
    }

    img_size = (50, 50)

    transform = [
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(means['imagenet'], stds['imagenet']),
    ]

    data_loader = data.DataLoader(
        dataset=datasets.ImageFolder(
            datas[case],
            transform=transforms.Compose(transform)
        ),
        num_workers=2,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    return data_loader
