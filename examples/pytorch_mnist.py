import torch
import torchvision
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
import argparse


parser = argparse.ArgumentParser(description='PYTORCH MNIST BENCHMARK')
parser.add_argument('--ckpt-path', default='/home/hzzheng/Code/faster-advt/TRADES/data-model/mnist.atta-1.mat/model-nn-epoch60.pt',
                    help='aaa')
parser.add_argument('-d', type=int, default=3,
                    help='Device number')
args = parser.parse_args()
# Setup the test loader
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:{}'.format(args.d))
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
transform_test = transforms.Compose([transforms.ToTensor(), ])
testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False, **kwargs)
test_dataset_array = next(iter(test_loader))[0].numpy()
test_label_dataset_array = next(iter(test_loader))[1].numpy()

batch_size = 512
epsilon = 0.3

if __name__ == '__main__':
    # print(use_cuda)
    ckpt_path = args.ckpt_path
    # Obtain the model object
    model = SmallCNN().to(device)

    # Define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    # Initialize the classifier
    mnist_classifier = PyTorchClassifier(clip_values=(0, 1), model=model, loss=criterion, optimizer=optimizer,
                                         input_shape=(1, 28, 28), nb_classes=10)

    # Load the classifier
    model.load_state_dict(torch.load(ckpt_path))

    # Test the classifier
    predictions = mnist_classifier.predict(test_dataset_array)

    accuracy = np.sum(np.argmax(predictions, axis=1) == test_label_dataset_array) / len(test_label_dataset_array)
    print('Accuracy before attack: {}%'.format(accuracy * 100))

    # Craft the adversarial examples

    # PGD-20
    # adv_crafter_pgd_40 = ProjectedGradientDescent(mnist_classifier, eps=epsilon, max_iter=40, batch_size=batch_size)
    #
    # x_test_adv = adv_crafter_pgd_40.generate(x=test_dataset_array)

    # Test the classifier on adversarial exmaples
    # predictions = mnist_classifier.predict(x_test_adv)
    # accuracy = np.sum(np.argmax(predictions, axis=1) == test_label_dataset_array) / len(test_label_dataset_array)
    # print('Accuracy after PGD-20 attack: {}%'.format(accuracy * 100))

    # PGD-100
    adv_crafter_pgd_100 = ProjectedGradientDescent(mnist_classifier, max_iter=100, batch_size=batch_size)

    x_test_adv = adv_crafter_pgd_100.generate(x=test_dataset_array)

    # Test the classifier on adversarial exmaples
    predictions = mnist_classifier.predict(x_test_adv)
    accuracy = np.sum(np.argmax(predictions, axis=1) == test_label_dataset_array) / len(test_label_dataset_array)
    print('Accuracy after PGD-100 attack: {}%'.format(accuracy * 100))

    # FGSM
    adv_crafter_fgsm = FastGradientMethod(mnist_classifier, eps=epsilon, batch_size=batch_size)
    x_test_adv = adv_crafter_fgsm.generate(x=test_dataset_array)

    # Test the classifier on adversarial exmaples
    predictions = mnist_classifier.predict(x_test_adv)
    accuracy = np.sum(np.argmax(predictions, axis=1) == test_label_dataset_array) / len(test_label_dataset_array)
    print('Accuracy after FGSM attack: {}%'.format(accuracy * 100))

    # DeepFool
    adv_crafter_deepfool = CarliniLInfMethod(mnist_classifier, batch_size=batch_size)
    x_test_adv = adv_crafter_deepfool.generate(x=test_dataset_array)

    predictions = mnist_classifier.predict(x_test_adv)
    accuracy = np.sum(np.argmax(predictions, axis=1) == test_label_dataset_array) / len(test_label_dataset_array)
    print('Accuracy after DeepFool attack: {}%'.format(accuracy * 100))

    # C&W

    adv_crafter_cwinf = CarliniLInfMethod(mnist_classifier, eps=epsilon, batch_size=batch_size)
    x_test_adv = adv_crafter_cwinf.generate(x=test_dataset_array)

    predictions = mnist_classifier.predict(x_test_adv)
    accuracy = np.sum(np.argmax(predictions, axis=1) == test_label_dataset_array) / len(test_label_dataset_array)
    print('Accuracy after C&W attack: {}%'.format(accuracy * 100))