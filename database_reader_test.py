import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os


def imshow(images, labels, classes):
    fig, axes = plt.subplots(1, len(images), figsize=(17, 4))
    for idx, (img, label) in enumerate(zip(images, labels)):
        npimg = img.numpy()
        axes[idx].imshow(np.transpose(npimg, (1, 2, 0)))
        axes[idx].set_title(classes[label])
        axes[idx].axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((61, 61)),

        ])

    train_dataset = datasets.ImageFolder(root='./database/figs/train', transform=transform)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)

    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    classes = train_dataset.classes

    imshow(images, labels, classes)






    

    

    
