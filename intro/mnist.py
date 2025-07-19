import torch
import torch.nn as nn
import torch.optim as optim
import torch_xla
import torch_xla.core.xla_model as xm
import torchvision
import torchvision.transforms as transforms

# Define a simple Convolutional Neural Network (CNN) for MNIST
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1) # Input channels: 1 (for grayscale MNIST)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(9216, 128) # (64 * 6 * 6) after convolutions and pooling
        self.relu3 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10) # Output classes: 10 (for MNIST digits 0-9)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1) # Flatten the tensor for the fully connected layers
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = self.log_softmax(x)
        return output

# --- Hyperparameters ---
lr = 0.01
momentum = 0.9
batch_size = 64
epochs = 5 # Number of training epochs

# --- XLA Device Setup ---
device = xm.xla_device()

# --- Data Loading and Preprocessing ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # Standard normalization for MNIST
])

# Download and load the MNIST training dataset
train_dataset = torchvision.datasets.MNIST(
    './data', train=True, download=True, transform=transform
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)

# Download and load the MNIST test dataset (for evaluation later, though not in this snippet)
test_dataset = torchvision.datasets.MNIST(
    './data', train=False, download=True, transform=transform
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
)

# --- Model, Optimizer, and Loss Function Initialization ---
model = Net().to(device) # Instantiate your defined CNN and move it to the XLA device
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
loss_fn = nn.NLLLoss()

# --- Training Loop ---
print(f"Starting training on device: {device}")
for epoch in range(1, epochs + 1):
    model.train() # Set the model to training mode
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()

        # Update model parameters and mark step for XLA
        xm.optimizer_step(optimizer) # Use xm.optimizer_step for XLA compatible optimizers
        # xm.mark_step() # xm.optimizer_step implicitly calls mark_step for the optimizer

        running_loss += loss.item()

        if batch_idx % 100 == 0: # Print every 100 batches
            print(f"Epoch: {epoch}/{epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {running_loss / (batch_idx + 1):.4f}")

    print(f"Epoch {epoch} finished. Average Loss: {running_loss / len(train_loader):.4f}")

print("Training complete!")

# You can add evaluation on the test set here if needed
# For example:
# model.eval()
# test_loss = 0
# correct = 0
# with torch.no_grad():
#     for data, target in test_loader:
#         data, target = data.to(device), target.to(device)
#         output = model(data)
#         test_loss += loss_fn(output, target).item()
#         pred = output.argmax(dim=1, keepdim=True)
#         correct += pred.eq(target.view_as(pred)).sum().item()

# xm.mark_step() # Mark step after evaluation for XLA to execute

# test_loss /= len(test_loader.dataset)
# print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
#       f'({100. * correct / len(test_loader.dataset):.0f}%)\n')