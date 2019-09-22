import argparse

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

parser = argparse.ArgumentParser(description='PYTORCH MNIST BENCHMARK')
parser.add_argument('--log-path',  default='./data-log/measure/atta-trades-atta-1.log',
                    help='Log path.')
parser.add_argument('--start-epoch', type=int, default=5,
                    help='The epoch number you start from.')
parser.add_argument('--total-epoch', type=int, default=20,
                    help='The number of epochs.')
parser.add_argument('--model-dir', default='./mnist.trades.atta-1.b6/',
                    help='The dir of the saved model')
parser.add_argument('--epsilon', type=int, default=0.3,
                    help='checkpoint')
parser.add_argument('--batch-size', type=int, default=512,
                    help='checkpoint')
args = parser.parse_args()


# Setup the test loader
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
transform_test = transforms.Compose([transforms.ToTensor(), ])
testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False, **kwargs)
test_dataset_array = next(iter(test_loader))[0].numpy()
test_label_dataset_array = next(iter(test_loader))[1].numpy()

# batch_size = 512
# log_path = "../data-log/mnist-trades-atta-1-accuracy.log"
# epsilon = 0.3
# start_epoch = 5;
# total_epoch = 20;
# directory_str = "../mnist.trades.atta-1.b6/"
# directory = os.fsencode(directory_str)
if __name__ == '__main__':
    log_file = open(args.log_path, 'w')

    # Obtain the model object
    model = SmallCNN().to(device)

    # Define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    # Initialize the classifier
    mnist_classifier = PyTorchClassifier(clip_values=(0, 1), model=model, loss=criterion, optimizer=optimizer,
                                         input_shape=(1, 28, 28), nb_classes=10)

    for epoch in range(args.total_epoch):
        e = (epoch + 1) * 5
        file = os.path.join(args.model_dir, "model-nn-epoch" + str(e) + ".pt")
        print(os.path.join(args.model_dir, "model-nn-epoch" + str(e) + ".pt"))
        # filename = os.fsdecode(file)
        # Load the classifier
        # print("Loading " + str(filename))
        model.load_state_dict(torch.load(file))

        # Test the classifier
        predictions = mnist_classifier.predict(test_dataset_array)

        accuracy = np.sum(np.argmax(predictions, axis=1) == test_label_dataset_array) / len(test_label_dataset_array)
        print('Accuracy before attack: {}%'.format(accuracy * 100))

        # Craft the adversarial examples

        # PGD-40
        adv_crafter_pgd_40 = ProjectedGradientDescent(mnist_classifier, eps=args.epsilon, max_iter=40, batch_size=args.batch_size)

        x_test_adv = adv_crafter_pgd_40.generate(x=test_dataset_array)

        # Test the classifier on adversarial exmaples
        predictions = mnist_classifier.predict(x_test_adv)
        accuracy = np.sum(np.argmax(predictions, axis=1) == test_label_dataset_array) / len(test_label_dataset_array)
        print('Accuracy after PGD-40 attack: {}%'.format(accuracy * 100))
        log_file.write("{} {}\n".format(e, accuracy))

    log_file.close()
