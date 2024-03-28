from natural_gradient_descent import NaturalGradientDescent
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch.optim as optim

def test_NaturalGradientDescent():
    # Define transforms to preprocess the data
    transform = transforms.Compose([
    transforms.ToTensor(),
    ])

    # Create datasets for training & validation, download if necessary
    training_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
    validation_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)

    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=64, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=64, shuffle=False)

    # Define a simple neural network model
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc1 = nn.Linear(784, 150, bias = True)
            self.fc2 = nn.Linear(150, 10, bias = True)

        def forward(self, x):
            x = torch.relu(self.fc1(x)) # Must use ReLU activation see pp.
            x = self.fc2(x)
           # print(x)
            return x

    # Create an instance of the 2-layer model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleModel().to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    parameters = list(model.parameters())
    optimizer = NaturalGradientDescent(parameters)
    #for p in parameters:
     #   print('p', p.shape)
    running_vloss = 0
    for i, vdata in enumerate(validation_loader):
        vinputs, vlabels = vdata
        vinputs = vinputs.view(vinputs.size(0), -1)  # Flatten the input images
        voutputs = model(vinputs)
        vloss = criterion(voutputs, vlabels)
        running_vloss += vloss

    def train_one_epoch(epoch_index, tb_writer):
        print('new epoch')
        running_loss = 0.
        last_loss = 0.

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
          #  print('weight size', model.fc1)
           # print('bias size', model.fc2)
          #  print('input size', inputs.size())
            outputs = model(inputs)
          #  print('outputs size', outputs.size())
          #  print('labels', labels)

            # Compute the loss and its gradients
            loss = criterion(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()
            # Gather data and report
            running_loss += loss.item()
            if i % 937 == 0:
                last_loss = running_loss / 937 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(training_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss
    
    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = 2

    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

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
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1

test_NaturalGradientDescent()