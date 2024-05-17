from datetime import datetime
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot
import os

from ComponentWiseNaturalGradientDescent import ComponentWiseNaturalGradientDescent

def test_NaturalGradientDescent():
    os.environ["PATH"] += os.pathsep + 'C:/Users/sammy/Downloads/windows_10_msbuild_Release_graphviz-10.0.1-win32.zip/Graphviz/bin'
    # Define transforms to preprocess the data
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # Create datasets for training & validation, download if necessary
    training_set = torchvision.datasets.FashionMNIST(
        "./data", train=True, transform=transform, download=True
    )
    validation_set = torchvision.datasets.FashionMNIST(
        "./data", train=False, transform=transform, download=True
    )

    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = torch.utils.data.DataLoader(
        training_set, batch_size=64, shuffle=True
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_set, batch_size=64, shuffle=False
    )

    # Define a simple neural network model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 150, bias=True)
            self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
            self.fc2 = nn.Linear(150, 10, bias=True)
            image, _ = training_set[0]
            self.image_size = image.size()

        def forward(self, x):
         #   x = x.view(-1, self.image_size[0], self.image_size[1], self.image_size[2])
            x = self.fc1(x)
            x = torch.relu(x)
            x = x.view(-1, self.image_size[0], x.shape[0], x.shape[1])
            x = self.conv1(x)
            x = torch.relu(x)
            x = self.fc2(x)
            x = x.view(x.shape[2], x.shape[3])
            return x
        
    # Define a list to store the linear outputs of the first layer
    activations = []

    # Create an instance of the 2-layer model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleModel().to(device)

    def hook(module, input, output):
        # Save output of the layer
        activations.append(output)

    # Registering hook to layer1, layer2
    unused_hook_handle = model.fc1.register_forward_hook(hook)
    unused_hook_handle2 = model.conv1.register_forward_hook(hook)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    parameters = list(model.parameters())

    # Create an OrderedDict to store the layers in order
    layers_dict = torch.nn.ModuleDict()

    # Iterate through the modules and add them to the OrderedDict
    for name, module in model.named_modules():
        if name != "":
            layers_dict[name] = module
    optimizer = ComponentWiseNaturalGradientDescent(parameters, activations, layers_dict, lr=0.01)

    running_vloss = 0
    for i, vdata in enumerate(validation_loader):
        vinputs, vlabels = vdata
        vinputs = vinputs.view(vinputs.size(0), -1)  # Flatten the input images
        voutputs = model(vinputs)
        vloss = criterion(voutputs, vlabels)
        running_vloss += vloss

    def train_one_epoch(epoch_index, tb_writer):
        running_loss = 0.0
        last_loss = 0.0

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(training_loader):
            # Every data instance is an input + label pair
            inputs, labels = data
            inputs = inputs.view(inputs.size(0), -1)  # Flatten the input images

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs)

            # Compute the loss and its gradients
            loss = criterion(outputs, labels)
            loss.backward(retain_graph = True)
            # Gradient Norm Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            gradients = torch.autograd.grad(loss, model.parameters(), retain_graph=True)
            # Adjust learning weights
            optimizer.step(gradients)
            # Gather data and report
            running_loss += loss.item()
            if i % 937 == 0:
                last_loss = running_loss / 937  # loss per batch
                print("  batch {} loss: {}".format(i + 1, last_loss))
                tb_x = epoch_index * len(training_loader) + i + 1
                tb_writer.add_scalar("Loss/train", last_loss, tb_x)
                running_loss = 0.0

        return last_loss

    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter("runs/fashion_trainer_{}".format(timestamp))
    epoch_number = 0

    EPOCHS = 2

    best_vloss = 1_000_000.0

    for _ in range(EPOCHS):
        print("EPOCH {}:".format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True) 
        avg_loss = train_one_epoch(epoch_number, writer)

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                vinputs = vinputs.view(vinputs.size(0), -1)  # Flatten the input images
                voutputs = model(vinputs)
                vloss = criterion(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print("LOSS train {} valid {}".format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars(
            "Training vs. Validation Loss",
            {"Training": avg_loss, "Validation": avg_vloss},
            epoch_number + 1,
        )
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = "model_{}_{}".format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1

test_NaturalGradientDescent()
