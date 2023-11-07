import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from NeuralNetwork import bag_of_words, tokenize, stem
from Brain import NeuralNet

# Load data from 'intents.json'
with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

# Populate 'all_words', 'tags', and 'xy' lists
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:
        print(pattern)
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = [',', '?', '/', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

x_train = []
y_train = []

# Prepare training data
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)

    label = tags.index(tag) if tag in tags else -1
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)

# Hyperparameters
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(x_train[0])
hidden_size = 8
output_size = len(tags)

print("Training The Model......")

# Define a custom dataset class
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset()

# Create a data loader
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

# Determine the device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the neural network model
model = NeuralNet(input_size, hidden_size, output_size).to(device=device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for words, label in train_loader:
        words = words.to(device)
        label = label.to(dtype=torch.long).to(device)
        outputs = model(words)
        loss = criterion(outputs, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        # Change this line to get the predicted class as a tensor
        predicted_class = torch.argmax(outputs, dim=1)

       # Print label and predicted_class tensors
    print(f"Label: {label}, Predicted: {predicted_class}")

# Check if any element in label tensor is greater than or equal to output_size
    if (label >= output_size).any():
        invalid_labels = label[label >= output_size]
        print(f"Invalid labels: {invalid_labels.tolist()}")


    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')



print(f'Final Loss: {epoch_loss:.4f}')

# Save the model and related data
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}
FILE = "Train_Data.pth"
torch.save(data, FILE)
print(f"Training complete. File saved to {FILE}")
