import numpy as np
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim


def load_data(batch=64):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])
    global trainset, valset, trainloader, valloader
    trainset = datasets.MNIST('MNIST/TRAINSET', download=True, train=True, transform=transform)
    valset = datasets.MNIST('MNIST/TESTSET', download=True, train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch, shuffle=True)


def show_examples_in_row():
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    plt.figure(figsize=(20, 4))
    for index, (image, label) in enumerate(zip(images[0:5], labels[0:5])):
        plt.subplot(1, 5, index + 1)
        plt.imshow(np.reshape(image, (28, 28)), cmap='gray_r')
        plt.title('Training: %i' % int(label), fontsize=20, color='black')
    plt.show()


def show_examples_in_table():
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    figure = plt.figure()
    num_of_images = 60
    for index in range(1, num_of_images + 1):
        plt.subplot(6, 10, index)
        plt.axis('off')
        plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
    plt.show()


def create_model():
    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10

    global model
    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[1], output_size),
                          nn.LogSoftmax(dim=1))
    print(model)


def train_model():
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    time0 = time()
    epochs = 1
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)

            # Training pass
            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)

            # This is where the model learns by backpropagating
            loss.backward()

            # And optimizes its weights here
            optimizer.step()

            running_loss += loss.item()
        else:
            print("Epoch {}/{} - Training loss: {}".format(e+1, epochs, running_loss / len(trainloader)))
    print("\nTraining Time (in minutes) =", (time() - time0) / 60)


def show_missed_prediction():
    wrong_image = None
    wrong_true_label = None
    wrong_pred_label = None
    for images, labels in valloader:
        find = False
        for i in range(len(labels)):
            img = images[i].view(1, 784)
            with torch.no_grad():
                logps = model(img)

            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]
            if true_label != pred_label:
                wrong_image = img
                wrong_true_label = int(true_label)
                wrong_pred_label = int(pred_label)
                find = True
                break
        if find:
            break

    plt.imshow(np.reshape(wrong_image, (28, 28)), cmap='gray_r')
    plt.title('Prediction: %i\nTrue label: %i\n' % (int(wrong_pred_label), int(wrong_true_label)), fontsize=20,
              color='black')
    plt.show()


def print_accuracy():
    correct_count, all_count = 0, 0
    for images, labels in valloader:
        for i in range(len(labels)):
            img = images[i].view(1, 784)
            with torch.no_grad():
                logps = model(img)

            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]
            if (true_label == pred_label):
                correct_count += 1
            all_count += 1
    print("Number Of Images Tested =", all_count)
    print("\nModel Accuracy =", (correct_count / all_count))


def try_on_image(image):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])
    input_image = Image.open(os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '../images/' + image)
    )

    rgbimage = np.asarray(input_image)
    gray_image = np.dot(rgbimage[..., :3], [0.2989, 0.5870, 0.1140])
    gray_image = transform(gray_image)

    with torch.no_grad():
        logps = model(gray_image)
        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        pred_label = probab.index(max(probab))

    plt.imshow(np.reshape(gray_image, (28, 28)), cmap='gray')
    plt.title('Prediction: %i\n' % int(pred_label), fontsize=20, color='black')
    plt.show()