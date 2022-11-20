import gen_exp
from gen_exp import expBuilderTest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable 
import numpy as np
import pickle

# For raytune
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

# Create experiment
seq_len = 100
img_size = 12
max_n_imgs = 2**img_size - 1
exp_generator = expBuilderTest(img_size = img_size, n_imgs = max_n_imgs, exp_size = max_n_imgs,
                                seq_len = seq_len, n_initial_view = 1, n_back = 1, p = 0.5 )

X_train, y_train, X_test, y_test = exp_generator.build_exp()

# Wrap train and test data in PyTorch tensors and Variables
X_train_tensors = Variable(torch.Tensor(X_train))
y_train_tensors = Variable(torch.Tensor(y_train))
X_test_tensors = Variable(torch.Tensor(X_test))
y_test_tensors = Variable(torch.Tensor(y_test)) 

print("Training Shape", X_train_tensors.shape, y_train_tensors.shape)
print("Testing Shape", X_test_tensors.shape, y_test_tensors.shape) 

'''
#reshaping to (len(data), seq_len, input_size)
X_train_tensors_final = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], X_train_tensors.shape[1], X_train_tensors.shape[1]))
X_test_tensors_final = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1])) 
y_train_tensors_final = torch.reshape(y_train_tensors, (y_train_tensors.shape[0], 1))
y_test_tensors_final = torch.reshape(y_test_tensors, (y_test_tensors.shape[0], 1)) 
'''
X_train_tensors_final = X_train_tensors
X_test_tensors_final = X_test_tensors
y_train_tensors_final = y_train_tensors
y_test_tensors_final = y_test_tensors
#y_train_tensors_final = torch.reshape(y_train_tensors, (y_train_tensors.shape[0], 1))
#y_test_tensors_final = torch.reshape(y_test_tensors, (y_test_tensors.shape[0], 1)) 

print("Training Shape", X_train_tensors_final.shape, y_train_tensors_final.shape)
print("Testing Shape", X_test_tensors_final.shape, y_test_tensors_final.shape) 

batch_size = 32
train_dataset = TensorDataset(X_train_tensors_final, y_train_tensors_final)
train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_dataset = TensorDataset(X_test_tensors_final, y_test_tensors_final)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        #self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        #self.relu = nn.ReLU()
        #self.fc = nn.Linear(128, seq_length) #fully connected last layer
        self.fc = nn.Linear(hidden_size, seq_length) #fully connected last layer
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        #print('hn shape:', hn.shape)
        #print('output shape:', output.shape)
        #output = output[:, -1, :]
        out = output[:, -1, :]
        #print('output reshaped:', output.shape)
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        #print('hn shape:', hn.shape)
        #out = self.relu(output)
        #print('relu shape:', out.shape)
        #out = self.fc_1(out) #first Dense
        #print('fc_1 shape:', out.shape)
        #out = self.relu(out) #relu
        #print('relu again shape:', out.shape)
        out = self.fc(out) #Final Output
        #print('final fc shape:', out.shape)
        return out

num_epochs = 2000
learning_rate = 0.01
#gamma = None
gamma = 0.992

input_size = img_size #number of features
hidden_size = 30 #number of features in hidden state
num_layers = 2 #number of stacked lstm layers
 
num_classes = 1 #number of output classes 

# Instantiate the LSTM
lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, X_train_tensors_final.shape[1]) #our lstm class 

#loss_fn = nn.BCEWithLogitsLoss()
loss_fn = nn.MultiLabelSoftMarginLoss()
optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma, verbose=True)

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/(y_test.shape[0]*y_test.shape[1])
    acc = torch.round(acc * 100)
    
    return acc

# Function to save the model 
def saveModel(model): 
    path = "tests/models/test_lstm_exp.pt" 
    torch.save(model.state_dict(), path)

best_acc = 0.0
save_obj = {'epoch' : [],
            'train loss' : [],
            'test loss' : [],
            'train accuracy' : [],
            'test accuracy' : [],
            'learning rate' : learning_rate,
            'gamma' : gamma if gamma else None,
            'hidden size' : hidden_size,
            'num_layers' : num_layers}
for epoch in range(num_epochs):
    lstm1.train()
    running_train_loss = 0.0 
    running_test_loss = 0.0
    running_train_acc = 0.0 
    running_test_acc = 0.0 
    train_n = 0
    test_n = 0
    for train_batch, train_labels in train_loader:
        train_out = lstm1.forward(train_batch) #forward pass
        optimizer.zero_grad() #calculate the gradient, manually setting to 0

        # Calculate train loss and accuracy
        train_loss = loss_fn(train_out, train_labels)
        running_train_loss += train_loss.item()
        train_n += train_out.size(0) 
        #train_acc = (torch.round(train_out) == y_train_tensors_final).sum().item()/train_n
        train_acc = binary_acc(train_out, train_labels)
        running_train_acc += train_acc.item()*train_batch.size(0)
        train_loss.backward() # backprop loss
        optimizer.step() #improve from loss, i.e backprop

    # Calculate test loss and accuracy
    with torch.no_grad():
        lstm1.eval()
        for test_batch, test_labels in test_loader:
            test_out = lstm1.forward(test_batch)
            test_loss = loss_fn(test_out, test_labels)
            running_test_loss += test_loss.item()
            test_n += test_out.size(0) 
            test_acc = binary_acc(test_out, test_labels)
            running_test_acc += test_acc.item()*test_batch.size(0)
    scheduler.step()
    
    # Calculate performance states for epoch
    train_loss_epoch = running_train_loss/len(train_loader) 
    test_loss_epoch = running_test_loss/len(test_loader)
    train_acc_epoch = (running_train_acc)/train_n
    test_acc_epoch = (running_test_acc)/test_n
    
    print("Epoch: %d, train loss: %1.5f, test loss: %1.5f, train acc: %1.5f, test acc: %1.5f" % (epoch, train_loss_epoch, test_loss_epoch, train_acc_epoch, test_acc_epoch)) 

    save_obj['epoch'].append(epoch)
    save_obj['train loss'].append(train_loss_epoch)
    save_obj['train accuracy'].append(train_acc_epoch)
    save_obj['test loss'].append(test_loss_epoch)
    save_obj['test accuracy'].append(test_acc_epoch)

    if test_acc_epoch > best_acc: 
        saveModel(lstm1)
        best_acc = test_acc_epoch 

    if epoch % 100 == 0:
        with open('tests/results/test_lstm_exp.pickle', 'wb') as handle:
            pickle.dump(save_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('\n')
print('Best accuracy:', best_acc)

'''
# Print some examples for sanity check!
with torch.no_grad():
    lstm1.eval()
    num_exs = 5
    for test_batch, test_labels in test_loader:
        test_out = lstm1.forward(test_batch)
        y_pred_tag = torch.round(torch.sigmoid(test_out))

        for idx in range(num_exs):
            print('Last two inputs for sequence:', idx)
            print(test_batch[idx][-2:])
            print('Target sequence:', idx)
            print(y_pred_tag[idx])
            print('\n')

        break
'''

