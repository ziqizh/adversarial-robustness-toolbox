import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

from collections import OrderedDict
from art.attacks.fast_gradient import FastGradientMethod
from art.attacks.deepfool import DeepFool
from art.attacks.adversarial_patch import AdversarialPatch
from art.attacks.hop_skip_jump import HopSkipJump
from art.attacks.carlini import CarliniL2Method
from art.attacks.projected_gradient_descent import ProjectedGradientDescent
from art.classifiers.pytorch import PyTorchClassifier
from art.utils import load_mnist

#Create the neural network architecture, return logits instead of activation in forward method (Eg. softmax).
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SmallCNN(nn.Module):
    def __init__(self, drop=0.5):
        super(SmallCNN, self).__init__()

        self.num_channels = 1
        self.num_labels = 10

        activ = nn.ReLU(True)

        self.feature_extractor = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.num_channels, 32, 3)),
            ('relu1', activ),
            ('conv2', nn.Conv2d(32, 32, 3)),
            ('relu2', activ),
            ('maxpool1', nn.MaxPool2d(2, 2)),
            ('conv3', nn.Conv2d(32, 64, 3)),
            ('relu3', activ),
            ('conv4', nn.Conv2d(64, 64, 3)),
            ('relu4', activ),
            ('maxpool2', nn.MaxPool2d(2, 2)),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(64 * 4 * 4, 200)),
            ('relu1', activ),
            ('drop', nn.Dropout(drop)),
            ('fc2', nn.Linear(200, 200)),
            ('relu2', activ),
            ('fc3', nn.Linear(200, self.num_labels)),
        ]))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.classifier.fc3.weight, 0)
        nn.init.constant_(self.classifier.fc3.bias, 0)

    def forward(self, input):
        features = self.feature_extractor(input)
        logits = self.classifier(features.view(-1, 64 * 4 * 4))
        return logits

# Setup the test loader
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
transform_test = transforms.Compose([transforms.ToTensor(),])
testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False, **kwargs)
test_dataset_array = next(iter(test_loader))[0].numpy()
test_label_dataset_array = next(iter(test_loader))[1].numpy()
print(test_label_dataset_array)

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_mnist()
x_train = np.swapaxes(x_train, 1, 3)
x_test = np.swapaxes(x_test, 1, 3)

# Obtain the model object
# device = torch.device("cuda")
model = SmallCNN().to(device)
# model = Net().to(device)

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# Initialize the classifier
mnist_classifier = PyTorchClassifier(clip_values=(0, 1), model=model, loss=criterion, optimizer=optimizer,
                                     input_shape=(1, 28, 28), nb_classes=10)

# Train the classifier
# mnist_classifier.fit(x_train, y_train, batch_size=64, nb_epochs=50)
# torch.save(model.state_dict(), "./minst.pt")
model.load_state_dict(torch.load("../checkpoints/model-nn-epoch100.pt"))


# Test the classifier
predictions = mnist_classifier.predict(test_dataset_array)
# print(predictions)

accuracy = np.sum(np.argmax(predictions, axis=1) == test_label_dataset_array) / len(y_test)
print('Accuracy before attack: {}%'.format(accuracy * 100))

# Craft the adversarial examples
epsilon = 0.2  # Maximum perturbation
# adv_crafter = AdversarialPatch(mnist_classifier, batch_size=16, max_iter=10)
# adv_crafter = FastGradientMethod(mnist_classifier, eps=epsilon)
# adv_crafter = CarliniL2Method(mnist_classifier)
adv_crafter = ProjectedGradientDescent(mnist_classifier)
# adv_crafter = DeepFool(mnist_classifier, epsilon=epsilon, max_iter=10)

x_test_adv = adv_crafter.generate(x=x_test)

# Test the classifier on adversarial exmaples
print(x_test_adv)
predictions = mnist_classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == test_label_dataset_array) / len(y_test)
print('Accuracy after attack: {}%'.format(accuracy * 100))