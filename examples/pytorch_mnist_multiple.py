import torch
import torchvision
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.wideresnet import *
from torchvision import datasets, transforms
import numpy as np

from collections import OrderedDict
from art.attacks.fast_gradient import FastGradientMethod
from art.attacks.deepfool import DeepFool
from art.attacks.adversarial_patch import AdversarialPatch
from art.attacks.hop_skip_jump import HopSkipJump
from art.attacks.carlini import CarliniLInfMethod
from models.small_cnn import *
from art.attacks.projected_gradient_descent import ProjectedGradientDescent
from art.classifiers.pytorch import PyTorchClassifier
from art.utils import load_mnist

# Setup the test loader
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
transform_test = transforms.Compose([transforms.ToTensor(), ])
testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False, **kwargs)
test_dataset_array = next(iter(test_loader))[0].numpy()
test_label_dataset_array = next(iter(test_loader))[1].numpy()

batch_size = 128
log_path = "../data-log/mnist-robust-accuracy.log"
epsilon = 0.3
start_epoch = 5;
total_epoch = 20;
directory_str = "../mnist.trades.atta-1.b6/"
directory = os.fsencode(directory_str)
log_file = open(log_path, 'w')

# Obtain the model object
model = SmallCNN().to(device)

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# Initialize the classifier
mnist_classifier = PyTorchClassifier(clip_values=(0, 1), model=model, loss=criterion, optimizer=optimizer,
                                     input_shape=(1, 28, 28), nb_classes=10)

for epoch in range(total_epoch):
    e = (epoch + 1) * 5
    file = os.path.join(directory_str, "model-nn-epoch" + str(e) + ".pt")
    print(os.path.join(directory_str, "model-nn-epoch" + str(e) + ".pt"))
    # filename = os.fsdecode(file)
    # Load the classifier
    # print("Loading " + str(filename))
    model.load_state_dict(torch.load(file))

    # Test the classifier
    predictions = mnist_classifier.predict(test_dataset_array)

    accuracy = np.sum(np.argmax(predictions, axis=1) == test_label_dataset_array) / len(test_label_dataset_array)
    print('Accuracy before attack: {}%'.format(accuracy * 100))

    # Craft the adversarial examples

    # PGD-20
    adv_crafter_pgd_40 = ProjectedGradientDescent(mnist_classifier, eps=epsilon, max_iter=40, batch_size=batch_size)

    x_test_adv = adv_crafter_pgd_40.generate(x=test_dataset_array)

    # Test the classifier on adversarial exmaples
    predictions = mnist_classifier.predict(x_test_adv)
    accuracy = np.sum(np.argmax(predictions, axis=1) == test_label_dataset_array) / len(test_label_dataset_array)
    print('Accuracy after PGD-20 attack: {}%'.format(accuracy * 100))
    log_file.write("{} {}\n".format(e, accuracy))

log_file.close()
