import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

df = pd.read_csv("data/data_with_features.csv")

df_features = df[["distance_miles", "time_of_day_hours", "is_weekend"]]
df_target = df["delivery_time_minutes"]

# conver data into tensors
distance = torch.tensor(df_features["distance_miles"], dtype=torch.float32)
time = torch.tensor(df_features["time_of_day_hours"], dtype=torch.float32)
week = torch.tensor(df_features["is_weekend"], dtype=torch.float32)

# normalize values of distance and time
distance_feature = (distance - distance.mean()) / distance.std()
time_feature = (time - time.mean()) / time.std()

# reshape the tensors fromn [100] to [1, 100]
distance_feature = distance_feature.unsqueeze(0)
time_feature = time_feature.unsqueeze(0)
week_feature = week.unsqueeze(0)

# combine all features for training set of shape [100, 1]
features = torch.cat((distance_feature, time_feature, week_feature), dim=0).transpose(0,1)

# get target tensor
target = torch.tensor(df_target, dtype=torch.float32).unsqueeze(1)

# define model
model = nn.Sequential(
    nn.Linear(3, 40),
    nn.ReLU(),
    nn.Linear(40, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)

# define hyperparameters
epochs = 200
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# define training loop
for i in tqdm(range(epochs), desc="Training", unit="epoch"):
    # reset gradients
    optimizer.zero_grad()
    # compute output
    prediction = model(features)
    # compute loss
    loss = loss_function(prediction, target)
    # compute gradients
    loss.backward()
    # update weights
    optimizer.step()

# Testing model with new instance
t_distance = 10.0
t_time = 9.5
test_weekend = 0.0

test_distance = (t_distance - distance.mean().item()) / distance.std().item()
test_time = (t_time - time.mean().item()) / time.std().item()
raw_input_tensor = torch.tensor([[test_distance, test_time, test_weekend]], dtype=torch.float32)
with torch.no_grad():
    prediction = model(raw_input_tensor)

# output the result
header = "+{:-<42}+{:-<23}+".format('', '')
title_line = "|{:^66}|".format(" Model Prediction ")
line1 = f"| {'Time of the Week':<40} | {f'{int(t_time)}:{int((t_time-int(t_time))*60)} hours':<21} |"
line2 = f"| {'Distance':<40} | {f'{t_distance:.1f} miles':<21} |"
line3 = f"| {'Weekend?':<40} | {test_weekend:<21} |"
line4 = f"| {'Prediction:':<40} | {f'{int(prediction.item())} minutes':<21} |"

print(header)
print(title_line)
print(header)
print(line1)
print(line2)
print(line3)
print(header)
print(line4)
print(header)