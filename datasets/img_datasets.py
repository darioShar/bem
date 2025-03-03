
import torch 
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.datasets import CIFAR10, CelebA, MNIST, ImageNet, ImageFolder
from .tinyimagenet import TinyImageNetDataset
from .lsun import LSUN
import torch.utils.data
from torch.utils.data import Subset, Dataset
import numpy as np
import os 
from PIL import Image
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )

def get_img_dataset(data_path, 
                    dataset_name,
                    image_size,
                    random_flip,
                    lsun_category=None,
                    ):
    
    is_discrete = False
    if dataset_name == 'binary_mnist':
        is_discrete = True
    
    
    if dataset_name == "cifar10":
        if random_flip is False:
            tran_transform = test_transform = transforms.Compose(
                [transforms.Resize(image_size), transforms.ToTensor()]
            )
        else:
            tran_transform = transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                ]
            )
        dataset = CIFAR10(
            os.path.join(data_path, "cifar10"),
            train=True,
            download=True,
            transform=tran_transform,
        )
        test_dataset = CIFAR10(
            os.path.join(data_path, "cifar10_test"),
            train=False,
            download=True,
            transform=tran_transform,#test_transform,
        )
    elif dataset_name == "cifar10_lt":
        if random_flip is False:
            tran_transform = test_transform = transforms.Compose(
                [transforms.Resize(image_size), transforms.ToTensor()]
            )
        else:
            tran_transform = transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                ]
            )
        # code taken from https://discuss.pytorch.org/t/how-to-implement-oversampling-in-cifar-10/16964/5
        # Load CIFAR10
        dataset = CIFAR10(
            root=os.path.join(data_path, "cifar10"),
            train=True,
            download=True,
            transform=tran_transform,
        )
        test_dataset = CIFAR10(
            root=os.path.join(data_path, "cifar10_test"),
            train=False,
            download=True,
            transform=tran_transform,#test_transform,
        )
        print('warning: does not introduce imbalance in the test dataset yet')
        # Get all training targets and count the number of class instances
        targets = np.array(dataset.targets)
        classes, class_counts = np.unique(targets, return_counts=True)
        nb_classes = len(classes)
        print('initial class count:', class_counts)

        # Create artificial imbalanced class counts
        # same as LIM 
        imbal_class_counts = [5000,2997,1796,1077,645,387,232,139,83,50]
        
        #[500, 5000] * 5

        # Get class indices
        class_indices = [np.where(targets == i)[0] for i in range(nb_classes)]

        # Get imbalanced number of instances
        imbal_class_indices = [class_idx[:class_count] for class_idx, class_count in zip(class_indices, imbal_class_counts)]
        imbal_class_indices = np.hstack(imbal_class_indices)

        # Set target and data to dataset
        dataset.targets = targets[imbal_class_indices]
        dataset.data = dataset.data[imbal_class_indices]

        # Get all training targets and count the number of class instances
        targets = np.array(dataset.targets)
        classes, class_counts = np.unique(targets, return_counts=True)
        nb_classes = len(classes)
        print('Final class count:', class_counts)

        assert len(dataset.targets) == len(dataset.data)
    elif dataset_name == "mini_cifar10":
        dataset = ImageFolder(
                root=os.path.join(data_path, "mini_cifar10"),
                transform = transforms.Compose(
                    [transforms.RandomHorizontalFlip(p=0.5),
                     transforms.ToTensor(),
                     transforms.Resize(image_size),
                     ]
                ),
            )
        test_dataset = ImageFolder(
                root=os.path.join(data_path, "mini_cifar10"),
                transform = transforms.Compose(
                    [transforms.RandomHorizontalFlip(p=0.5),
                     transforms.ToTensor(),
                     transforms.Resize(image_size),
                     ]
                ),
            )
    
    elif dataset_name == "mnist" or dataset_name == "binary_mnist":
        mnist_transforms = [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        if dataset_name == "binary_mnist":
            mnist_transforms.append(transforms.Lambda(lambda x: x > 0.5))
            # transform to float
            mnist_transforms.append(transforms.Lambda(lambda x: x.float()))
        mnist_transform = transforms.Compose(mnist_transforms)

        dataset = MNIST(
            os.path.join(data_path, "mnist"),
            train=True,
            download=True,
            transform=mnist_transform,
        )
        test_dataset = MNIST(
            os.path.join(data_path, "mnist_test"),
            train=False,
            download=True,
            transform=mnist_transform, #test_transform,
        )
        
    elif dataset_name == "celeba":
        cx = 89
        cy = 121
        x1 = cy - 64
        x2 = cy + 64
        y1 = cx - 64
        y2 = cx + 64
        if random_flip:
            dataset = CelebA(
                root=os.path.join(data_path),#, "celeba"),
                split="train",
                transform=transforms.Compose(
                    [
                        Crop(x1, x2, y1, y2),
                        transforms.Resize(image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        
                    ]
                ),
                download=True,
            )
        else:
            dataset = CelebA(
                root=os.path.join(data_path),#, "celeba"),
                split="train",
                transform=transforms.Compose(
                    [
                        Crop(x1, x2, y1, y2),
                        transforms.Resize(image_size),
                        transforms.ToTensor(),
                        
                    ]
                ),
                download=True,
            )

        test_dataset = CelebA(
            root=os.path.join(data_path),#, "celeba"),
            split="test",
            transform=transforms.Compose(
                [
                    Crop(x1, x2, y1, y2),
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    
                ]
            ),
            download=True,
        )
    
    elif dataset_name == "celeba_hq":
        if random_flip:
            dataset = ImageFolder(
                root=os.path.join(data_path, "celebahq"),
                transform = transforms.Compose(
                    [transforms.RandomHorizontalFlip(p=0.5),
                     transforms.ToTensor(),
                     transforms.Resize(image_size),
                     ]
                ),
            )
        else:
            dataset = ImageFolder(
                root=os.path.join(data_path, "celebahq"),
                transform = transforms.Compose(
                    [transforms.ToTensor(),
                     transforms.Resize(image_size),
                     ]
                ),
            )

        num_items = len(dataset)
        indices = list(range(num_items))
        random_state = np.random.get_state()
        np.random.seed(2019)
        np.random.shuffle(indices)
        np.random.set_state(random_state)
        train_indices, test_indices = (
            indices[: int(num_items * 0.9)],
            indices[int(num_items * 0.9):],
        )
        test_dataset = Subset(dataset, test_indices)
        dataset = Subset(dataset, train_indices)        

    elif dataset_name == "lsun":
        train_folder = "{}_train".format(lsun_category)
        val_folder = "{}_val".format(lsun_category)
        if random_flip:
            dataset = LSUN(
                root=os.path.join(data_path, "lsun"),
                classes=[train_folder],
                transform=transforms.Compose(
                    [
                        transforms.Resize(image_size),
                        transforms.CenterCrop(image_size),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.ToTensor(),
                        
                    ]
                ),
            )
        else:
            dataset = LSUN(
                root=os.path.join(data_path, "lsun"),
                classes=[train_folder],
                transform=transforms.Compose(
                    [
                        transforms.Resize(image_size),
                        transforms.CenterCrop(image_size),
                        transforms.ToTensor(),
                        
                    ]
                ),
            )

        test_dataset = LSUN(
            root=os.path.join(data_path, "lsun"),
            classes=[val_folder],
            transform=transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    
                ]
            ),
        )

    elif dataset_name == "ffhq":
        if random_flip:
            dataset = FFHQ(
                path=os.path.join(data_path, "FFHQ"),
                transform=transforms.Compose(
                    [transforms.RandomHorizontalFlip(p=0.5), 
                     transforms.ToTensor(), 
                     ]
                ),
                resolution=image_size,
            )
        else:
            dataset = FFHQ(
                path=os.path.join(data_path, "FFHQ"),
                transform=transforms.ToTensor(),
                resolution=image_size,
            )

        num_items = len(dataset)
        indices = list(range(num_items))
        random_state = np.random.get_state()
        np.random.seed(2019)
        np.random.shuffle(indices)
        np.random.set_state(random_state)
        train_indices, test_indices = (
            indices[: int(num_items * 0.9)],
            indices[int(num_items * 0.9) :],
        )
        test_dataset = Subset(dataset, test_indices)
        dataset = Subset(dataset, train_indices)
    elif dataset_name == 'tinyimagenet':
        if random_flip:
            transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Resize(image_size),
                    ]
            )
        else:
            transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Resize(image_size),
                ]

            )

        dataset = TinyImageNetDataset(root_dir=os.path.join(data_path, 'tiny-imagenet-200'), 
                                      mode='train', 
                                      transform=transform,
                                      download=False,
                                      preload=False)
        test_dataset = TinyImageNetDataset(root_dir=os.path.join(data_path, 'tiny-imagenet-200'), 
                                      mode='test', 
                                      transform=transform,
                                      download=False,
                                      preload=False)
        #train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    else:
        print("Image dataset {} not found. Returning None datasets...".format(dataset_name))
        dataset, test_dataset = None, None

    return dataset, test_dataset, is_discrete


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = (2. * X - 1.0)*5.
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, "image_mean"):
        return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config, X):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 5.0) / 10.0

    return torch.clamp(X, 0.0, 1.0)

class imagenet64_dataset(Dataset):
    """`DownsampleImageNet`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    train_list = [
        ['train_data_batch_1'],
        ['train_data_batch_2'],
        ['train_data_batch_3'],
        ['train_data_batch_4'],
        ['train_data_batch_5'],
        ['train_data_batch_6'],
        ['train_data_batch_7'],
        ['train_data_batch_8'],
        ['train_data_batch_9'],
        ['train_data_batch_10']
    ]
    test_list = [
        ['val_data'],
    ]

    def __init__(self, root, train=True):
        self.root = os.path.expanduser(root)
        self.transform = transforms.ToTensor()
        self.train = train  # training set or test set

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()
            # resize label range from [1,1000] to [0,1000),
            # This is required by CrossEntropyLoss
            self.train_labels[:] = [x - 1 for x in self.train_labels]

            self.train_data = np.concatenate(self.train_data)
            [picnum, pixel] = self.train_data.shape
            pixel = int(np.sqrt(pixel / 3))
            self.train_data = self.train_data.reshape((picnum, 3, pixel, pixel))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            [picnum,pixel]= self.test_data.shape
            pixel = int(np.sqrt(pixel/3))

            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()

            # resize label range from [1,1000] to [0,1000),
            # This is required by CrossEntropyLoss
            self.test_labels[:] = [x - 1 for x in self.test_labels]
            self.test_data = self.test_data.reshape((picnum, 3, pixel, pixel))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, y_data = self.train_data[index], self.train_labels[index]
        else:
            img, y_data = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        x_data = self.transform(img)
        y_data = torch.tensor(y_data, dtype=torch.int64)

        return x_data, y_data

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)