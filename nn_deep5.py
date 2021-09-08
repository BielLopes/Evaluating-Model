import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 5
hidden_size = 25
num_epochs = 3000
data_tresholder = 205
test_size = 51
data_source = 0
batch_size = 105
output_size = 1
learning_rate = 0.001

# Create Dataset Model
class NandGateDataset(Dataset):
    
    def __init__(self, transform=None, is_test=False, tresholder=0, data_source=0):
        # Data loading
        if (data_source == 0):
            bd = pd.read_csv('complete_data.csv').to_numpy()
        if (data_source == 1):
            bd = pd.read_csv('algSolution_data.csv').to_numpy()
        if (data_source == 2):
            bd = pd.read_csv('algSolution_data2.csv').to_numpy()
        if (data_source == 3):
            bd = pd.read_csv('algSolution_data3.csv').to_numpy()
        
        self.n_samples = bd.shape[0]

        if is_test:
            self.x = bd[tresholder:,:5]
            self.y = bd[tresholder:,[5]] # n_samples, 1
            self.n_samples = bd.shape[0] - tresholder
        else:
            if tresholder:
                self.x = bd[:tresholder,:5]
                self.y = bd[:tresholder,[5]] # n_samples, 1
                self.n_samples = tresholder
            else:
                self.x = bd[:,:5]
                self.y = bd[:,[5]] # n_samples, 1
                self.n_samples = bd.shape[0]
                

        self.transform = transform


    def __getitem__(self, index):
        # dataset[index]
        sample =  self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        # len(dataset)
        return self.n_samples

class ToTensor:
    # Convert ndarrays to Tensors
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs).type(torch.float32), torch.from_numpy(targets).type(torch.float32)

class OutputTransform:
    # multiply inputs with a given factor
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, targets = sample
        targets *= self.factor
        return inputs, targets


composed_transform = transforms.Compose([ToTensor(), OutputTransform(10e12)])
train_data = NandGateDataset(transform=composed_transform, tresholder=data_tresholder)
test_data = NandGateDataset(transform=composed_transform, is_test=True, tresholder=data_tresholder)

train_loader = DataLoader(dataset=train_data,
                         batch_size=batch_size,
                         shuffle=True)

test_loader = DataLoader(dataset=test_data,
                         batch_size=batch_size,
                         shuffle=True)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.relu = nn.functional.leaky_relu
        self.li = nn.Linear(input_size, hidden_size) 
        self.lh2 = nn.Linear(hidden_size, hidden_size)
        self.lh3 = nn.Linear(hidden_size, hidden_size)
        self.lh4 = nn.Linear(hidden_size, hidden_size)
        self.lh5 = nn.Linear(hidden_size, hidden_size)
        self.lo = nn.Linear(hidden_size, output_size)  
    
    def forward(self, x):
        out = self.li(x)
        out = self.relu(out)
        out = self.lh2(out)
        out = self.relu(out)
        out = self.lh3(out)
        out = self.relu(out)
        out = self.lh4(out)
        out = self.relu(out)
        out = self.lh5(out)
        out = self.relu(out)
        out = self.lo(out)
        # no activation and no softmax at the end
        return out

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (cur_data, labels) in enumerate(train_loader):  
        # Load on device
        cur_data = cur_data.to(device)
        labels = labels.to(device)
        
        # Forward pass
        output = model(cur_data)
        loss = criterion(output, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 1000 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')


# Test the model
with torch.no_grad():
    sum_error_loss = 0
    for cur_data, labels in test_loader:
        cur_data = cur_data.to(device)
        labels = labels.to(device)
        output = model(cur_data)
        sum_error_loss += criterion(output, labels)

    print(f'Mean error inside train dataset: {sum_error_loss/test_size}')