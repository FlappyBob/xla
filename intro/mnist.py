import torch
import torch.nn as nn
import torch.optim as optim
import torch_xla
import torch_xla.core.xla_model as xm
import torchvision
import torchvision.transforms as transforms
from tqdm.auto import tqdm # Import tqdm for progress bars

from model import Net

# --- Hyperparameters ---
lr = 0.01
momentum = 0.9
batch_size = 128
epochs = 5
print_interval = 100 # Print detailed loss every N batches

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

# --- Device Selection ---
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
try:
    # Try to get an XLA device first (most robust way)
    xla_device = xm.xla_device()
    device = xla_device
    print("XLA device detected.")
except Exception:
    # If XLA is not available, then the device remains 'cuda' (if available) or 'cpu'
    if device.type == 'cuda':
        print("CUDA (GPU) device detected.")
    else:
        print("No CUDA or XLA device found, falling back to CPU.")

print(f"Using device: {device}")

# --- Model, Optimizer, and Loss Function Initialization ---
model = Net().train().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
loss_fn = nn.NLLLoss()

# --- Training Loop ---
print(f"Starting training on device: {device}")
for epoch in range(1, epochs + 1):
    model.train() # Set the model to training mode
    running_loss = 0.0
    
    # Wrap train_loader with tqdm for a progress bar
    # set_postfix allows updating metrics next to the bar
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch")

    for batch_idx, (data, target) in enumerate(train_bar):
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()

        optimizer.step()
        xm.mark_step()
        # Accumulate loss (this is a device-to-host transfer if loss is on XLA)
        # For XLA, it's better to accumulate on device and reduce later if possible
        # but for simple scalar loss, .item() is common.
        current_loss = loss.item() 
        running_loss += current_loss

        # Update tqdm postfix with current loss
        train_bar.set_postfix(loss=f"{current_loss:.4f}")

        # You can still print periodically for more detailed logs
        if (batch_idx + 1) % print_interval == 0:
            print(f"Epoch: {epoch}/{epochs}, Batch: {batch_idx+1}/{len(train_loader)}, Loss: {current_loss:.4f}")

    avg_epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch} finished. Average Loss: {avg_epoch_loss:.4f}")

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