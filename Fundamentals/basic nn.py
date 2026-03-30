import torch
import torch.nn as nn
import torch.optim as opt
from tqdm import tqdm

distances = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
times = torch.tensor([[6.96], [12.11], [16.77], [22.21]], dtype=torch.float32)

# Define model
model = nn.Sequential(nn.Linear(1, 1)) # Define a sequential model with 1 input and 1 output

# Define loss function
loss_fn = nn.MSELoss()

# Define optimization algorithm
optim = opt.SGD(model.parameters(), lr=0.01)

# Define training epochs
epochs = 50

# Define training cycle
for i in tqdm(range(epochs), desc="Training"):
    # Reset gradients
    optim.zero_grad
    # Calculate outputs
    outputs = model(distances)
    # Calculate loss base on the outputs
    loss = loss_fn(outputs, times)
    # Calculate gradients
    loss.backward()
    # Update weights
    optim.step()

# Inference on new instance
new_sample = torch.tensor([[7.0]], dtype=torch.float32)
with torch.no_grad():
    prediction = model(new_sample)
    print(f"For a distnce of: {new_sample.item()}, estimated time delivery: {prediction.item()}")

    # Decision boundary
    if prediction.item() > 30:
        print("DO NOT! take the job!")
    else:
        print("ALL IN! take the job!")