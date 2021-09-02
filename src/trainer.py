import torch.nn.functional as F
from src.loadingdata import train_loader
import torch.optim as optim
from src.model import CNN1
from src.loadingdata import input_size, output_size, get_n_params


# This function trains the neural network for one epoch
def train(epoch, model):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        output = model(data)  # Forward pass of the neural net
        optimizer.zero_grad(set_to_none=False)  # Zero out gradients from previous step
        loss = F.nll_loss(output, target)   # Calculation of the loss function
        loss.backward()  # Backward pass (gradient computation)
        optimizer.step()   # Adjusting the parameters according to the loss function

        if batch_idx % 10 and batch_idx > 5:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]   \tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


# Training settings  for model 1
n_features = 2  # hyper parameter
model1 = CNN1(input_size, n_features, output_size)
optimizer = optim.SGD(model1.parameters(), lr=0.01, momentum=0.5)
print('Number of parameters: {}'.format(get_n_params(model1)))