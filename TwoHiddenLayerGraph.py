import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
num_epochs = 400
batch_size = 105
output_size = 1
learning_rate = 0.001

folder_list = [(0, 26), (26, 51), (51, 77), (77, 102), (102, 128), (128, 154), (154, 179), (179, 205), (205,230), (230, 256)]

# Create Dataset Model
class NandGateDataset(Dataset):
    
    def __init__(self, transform=None, is_test=False, folder_numbers=[0,1]):
        # Data loading
        data_source = pd.read_csv('complete_data.csv').to_numpy()
        min_values = []
        max_values = []
        for i in range(6):
            min_values.append(data_source[:,i].min())
        
        for i in range(6):
            data_source[:,i] += -min_values[i]
            max_values.append(data_source[:,i].max())

        first_folder = True
        if is_test:
            for i in range(10):
                if i in folder_numbers:
                    if first_folder:
                        self.x = data_source[folder_list[i][0]:folder_list[i][1],:5]
                        self.y = data_source[folder_list[i][0]:folder_list[i][1],[5]]
                        first_folder = False
                    else:
                        self.x = np.concatenate((self.x, data_source[folder_list[i][0]:folder_list[i][1],:5]))
                        self.y = np.concatenate((self.y, data_source[folder_list[i][0]:folder_list[i][1],[5]]))
        else:
            for i in range(10):
                if i not in folder_numbers:
                    if first_folder:
                        self.x = data_source[folder_list[i][0]:folder_list[i][1],:5]
                        self.y = data_source[folder_list[i][0]:folder_list[i][1],[5]]
                        first_folder = False
                    else:
                        self.x = np.concatenate((self.x, data_source[folder_list[i][0]:folder_list[i][1],:5]))
                        self.y = np.concatenate((self.y, data_source[folder_list[i][0]:folder_list[i][1],[5]]))
        
        self.n_samples= len(self.x)

        self.x = self.x/max_values[0:5]
        self.y = self.y/max_values[5]

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

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size_one, hidden_size_two, output_size):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size_one) 
        self.activation = nn.functional.leaky_relu
        self.l2 = nn.Linear(hidden_size_one, hidden_size_two)
        self.l3 = nn.Linear(hidden_size_two, output_size)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.activation(out)
        out = self.l2(out)
        out = self.activation(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out

# Tudo indica que aqui começa o loop de avaliação
initial_folder_numbers=[0,1]

for plus_first_hidden_size in range(11):
    
    input_size = 5
    hidden_size_one = 5 + plus_first_hidden_size

    for plus_second_hidden_size in range(3):
        
        hidden_size_two = 2 + plus_second_hidden_size

        save_folder = f'Hidden_{hidden_size_one}_{hidden_size_two}'
        parent_dir = "/home/petrobras/Documents/Evaluating Model"
        path = os.path.join(parent_dir, save_folder)
        if not os.path.exists(path):
            os.mkdir(path)
            
        for k in range(9):
            loop_folder_numbers = [x+k for x in initial_folder_numbers]
            # print(f"New folder validating: {loop_folder_numbers}")

            train_data = NandGateDataset(transform=ToTensor(), folder_numbers=loop_folder_numbers)
            test_data = NandGateDataset(transform=ToTensor(), is_test=True, folder_numbers=loop_folder_numbers)

            train_loader = DataLoader(dataset=train_data,
                                    batch_size=batch_size,
                                    shuffle=True)

            test_loader = DataLoader(dataset=test_data,
                                    batch_size=int(batch_size/20),
                                    shuffle=True)

            model = NeuralNet(input_size, hidden_size_one, hidden_size_two, output_size).to(device)

            # Loss and optimizer
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

            # Train the model
            error_training = []
            error_test = []

            for epoch in range(num_epochs):
                partial_loss = []
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
                    
                    partial_loss.append(loss.item())

                error_training.append(sum(partial_loss)/len(partial_loss))
                    
                error_loss = np.array([])

                with torch.no_grad():
                    for cur_data, labels in test_loader:
                        cur_data = cur_data.to(device)
                        labels = labels.to(device)
                        output = model(cur_data)
                        loss = criterion(output, labels)
                        error_loss = np.append(error_loss, loss)

                error_test.append(np.sum(error_loss)/len(error_loss))

            plt.plot(error_training, color='red', label='Error of Training')
            plt.plot(error_test, color='blue', label='Error of Test')
            plt.xlabel('Epochs')
            plt.ylabel('Mean Error')
            plt.title(f'Hidden Sizes: {hidden_size_one}, {hidden_size_two}; Folders: {loop_folder_numbers[0]}, {loop_folder_numbers[1]} ')
            plt.legend()
            filename = f'folders_{loop_folder_numbers[0]}_{loop_folder_numbers[1]}.png'
            plt.savefig(f'{save_folder}/{filename}')
            plt.close()

    print("#############################################") 
