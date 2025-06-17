import gdown
import os
from torchvision import transforms
import matplotlib.pyplot as plt
from data.multimnist_dataloader import MNISTLoader

def load_data(file_id):
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'multi_mnist.pickle'
    if not os.path.exists(output):
        print("Downloading dataset...")
        gdown.download(url, output, quiet=False)
    else:
        print(f"Dataset already exists at '{output}', skipping download.")

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ])

    data = MNISTLoader(batch_size=256,
                    train_transform=transform,
                    test_transform=transform,
                    file_path='multi_mnist.pickle')

    dat = next(iter(data.train_dataloader()))
    print(dat[0].shape)
    ims = dat[0] # Tensor (batch_size, 1, 36, 36)
    labs_l = dat[1][:, 0]
    labs_r = dat[1][:, 1]
    f, axarr = plt.subplots(4, 8, figsize=(20, 10))
    for j in range(8):
        for i in range(4):
            axarr[i][j].imshow(ims[j*2+i].squeeze(0).numpy(), cmap='gray') 
            axarr[i][j].set_title('{}_{}'.format(labs_l[j*2+i],labs_r[j*2+i]))
    plt.tight_layout()
    plt.show()
    return data

    