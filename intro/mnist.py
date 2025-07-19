import torch
import torch.nn as nn
import torch.optim as optim
import torch_xla
import torch_xla.core.xla_model as xm
import torchvision
import torchvision.transforms as transforms

from mnist import Net

# --- Hyperparameters ---
lr = 0.01
momentum = 0.9
batch_size = 64
epochs = 5

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


device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
if xm.xla_device():
    device = xm.xla_device()

print(f"Using device: {device}")

model = Net().train().to(device) 
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