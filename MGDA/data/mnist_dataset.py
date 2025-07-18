import torch
from PIL import Image
import os
import numpy as np
import codecs
from skimage.transform import resize


class MNIST(torch.utils.data.Dataset):
    """MNIST Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
        'https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
        'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
        'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'multi_training.pt'
    test_file = 'multi_test.pt'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        
        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found. Use download=True to download it.')

        file_name = self.training_file if self.train else self.test_file
        self.data, (self.labels_l, self.labels_r) = torch.load(
            os.path.join(self.root, self.processed_folder, file_name)
        )


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = self.data[index]
        target_l_and_target_r = (self.labels_l[index], self.labels_r[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy().astype(np.uint8), mode='L')
        if self.transform is not None:
            img = self.transform(img)

        return img, target_l_and_target_r #target_l, target_r

    def __len__(self):
        return len(self.data)
    
    def _check_exists(self):
        return  os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        os.makedirs(os.path.join(self.root, self.raw_folder), exist_ok=True)
        os.makedirs(os.path.join(self.root, self.processed_folder), exist_ok=True)

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')
        multi_mnist_ims, extension = read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte'))
        multi_mnist_labels_l, multi_mnist_labels_r = read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'), extension)

        tmulti_mnist_ims, textension = read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte'))
        tmulti_mnist_labels_l, tmulti_mnist_labels_r = read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'), textension)


        multi_mnist_training_set = (multi_mnist_ims, (multi_mnist_labels_l, multi_mnist_labels_r))
        multi_mnist_test_set = (tmulti_mnist_ims, (tmulti_mnist_labels_l, tmulti_mnist_labels_r))

        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(multi_mnist_training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(multi_mnist_test_set, f)
        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path, extension):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        multi_labels_l = np.zeros((length),dtype=np.long)
        multi_labels_r = np.zeros((length),dtype=np.long)
        for im_id in range(length):
            for rim in range(1):
                multi_labels_l[im_id+rim] = parsed[im_id]
                multi_labels_r[im_id+rim] = parsed[extension[im_id+rim]] 
        return torch.from_numpy(multi_labels_l).view(length*1).long(), torch.from_numpy(multi_labels_r).view(length*1).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        pv = parsed.reshape(length, num_rows, num_cols)

        multi_data = np.zeros((length, num_rows, num_cols))
        extension = np.zeros(length, dtype=np.int32)

        for left in range(length):
            chosen_ones = np.random.permutation(length)[:1]
            extension[left*1:(left+1)*1] = chosen_ones
            for j, right in enumerate(chosen_ones):
                lim = pv[left,:,:]
                rim = pv[right,:,:]
                new_im = np.zeros((36,36))
                new_im[0:28,0:28] = lim
                new_im[6:34,6:34] = rim
                new_im[6:28,6:28] = np.maximum(lim[6:28,6:28], rim[0:22,0:22])
                multi_data_im = resize(new_im, (28, 28))  #m.imresize(new_im, (28, 28), interp='nearest')
                multi_data[left*1 + j,:,:] = multi_data_im
        return torch.from_numpy(multi_data).view(length,num_rows, num_cols), extension