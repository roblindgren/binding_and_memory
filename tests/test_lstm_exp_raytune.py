import gen_exp_val
from gen_exp_val import expBuilderTest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable 
import numpy as np
import pickle

# For raytune
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
#from ray.air import session
#from ray.air.checkpoint import Checkpoint
import os
from functools import partial


def gen_data(seq_len, img_size,):
    
    # Create experiment
    max_n_imgs = 2**img_size - 1
    exp_generator = expBuilderTest(img_size = img_size, n_imgs = max_n_imgs, exp_size = max_n_imgs,
                                    seq_len = seq_len, n_initial_view = 1, n_back = 1, p = 0.5 )

    X_train, y_train, X_test, y_test, X_hold, y_hold = exp_generator.build_exp()

    # Wrap train and test data in PyTorch tensors and Variables
    X_train_tensors = Variable(torch.Tensor(X_train))
    y_train_tensors = Variable(torch.Tensor(y_train))
    X_test_tensors = Variable(torch.Tensor(X_test))
    y_test_tensors = Variable(torch.Tensor(y_test)) 
    X_hold_tensors = Variable(torch.Tensor(X_hold))
    y_hold_tensors = Variable(torch.Tensor(y_hold)) 

    X_train_tensors_final = X_train_tensors
    X_test_tensors_final = X_test_tensors
    X_hold_tensors_final = X_hold_tensors
    y_train_tensors_final = y_train_tensors
    y_test_tensors_final = y_test_tensors
    y_hold_tensors_final = y_hold_tensors
    

    print("Training Shape", X_train_tensors_final.shape, y_train_tensors_final.shape)
    print("Testing Shape", X_test_tensors_final.shape, y_test_tensors_final.shape) 
    print("Holdout Shape", X_hold_tensors_final.shape, y_hold_tensors_final.shape) 

    train_dataset = TensorDataset(X_train_tensors_final, y_train_tensors_final)
    test_dataset = TensorDataset(X_test_tensors_final, y_test_tensors_final)
    hold_dataset = TensorDataset(X_hold_tensors_final, y_hold_tensors_final)

    return train_dataset, test_dataset, hold_dataset

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

def train_raytune(config, LSTM1, input_size, seq_len, train_dataset, test_dataset, num_epochs=10, checkpoint_dir=None):
    # Instantiate the LSTM
    lstm1 = LSTM1(1, input_size, config['hidden_size'], config['num_layers'], seq_len) #our lstm class 

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            lstm1 = nn.DataParallel(lstm1)
    lstm1.to(device)

    loss_fn = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(lstm1.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['gamma'], verbose=True)

    if checkpoint_dir:
        print('*************CHECKPOINT IT*************** \n')
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        lstm1.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])

    for epoch in range(num_epochs):
        lstm1.train()
        running_train_loss = 0.0 
        running_test_loss = 0.0
        running_train_acc = 0.0 
        running_test_acc = 0.0 
        train_n = 0
        test_n = 0
        for train_batch, train_labels in train_loader:
            train_batch, train_labels = train_batch.to(device), train_labels.to(device)
            
            # zero the parameter gradient
            optimizer.zero_grad() #calculate the gradient, manually setting to 0

            # Forward, backward, and optimize
            train_out = lstm1.forward(train_batch) #forward pass
            train_loss = loss_fn(train_out, train_labels)
            train_loss.backward() # backprop loss
            optimizer.step() #improve from loss, i.e backprop

            # calc train stats
            running_train_loss += train_loss.item()
            train_n += train_out.size(0) 
            train_acc = binary_acc(train_out, train_labels)
            running_train_acc += train_acc.item()*train_batch.size(0)

        # Calculate test loss and accuracy
        with torch.no_grad():
            lstm1.eval()
            for test_batch, test_labels in test_loader:
                test_batch, test_labels  = test_batch.to(device), test_labels.to(device)

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

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((lstm1.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=test_loss_epoch, accuracy=test_acc_epoch)
        

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/(y_test.shape[0]*y_test.shape[1])
    acc = torch.round(acc * 100)
    
    return acc

# Function to save the model 
def saveModel(model, path): 
    torch.save(model.state_dict(), path)

def main():    
    # Set directories
    data_dir = os.path.abspath("./tests/raytune_data/")
    checkpoint_dir = os.path.abspath("./tests/raytune_checkpoint/")
    print('checkpoint_dir is:', checkpoint_dir)

    # Task parameters 
    num_classes = 1 
    
    # Experiment parameters
    seq_len = 100
    img_size = 12

    # Model parameters
    input_size = img_size
    
    # Tune parameters
    num_samples=3
    max_num_epochs=500
    gpus_per_trial=0
    cpus_per_trial=1

    #load_data(data_dir)

    config = {
        "hidden_size": tune.grid_search([100, 110, 125, 150]),
        "num_layers": tune.grid_search([1, 2]),
        "lr": tune.grid_search([1e-6, 1e-5, 1e-4, 1e-3]),
        "batch_size": tune.grid_search([32, 64, 128]),
        "gamma": tune.grid_search([0.9, 0.99, 0.999])
    }

    train_dataset, val_dataset, test_dataset = gen_data(seq_len, img_size)

    ray_scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=10,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["hidden_size", "num_layers", "lr", "batch_size", "gamma"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        partial(train_raytune, LSTM1=LSTM1, input_size=input_size, seq_len=seq_len, num_epochs=max_num_epochs, train_dataset=train_dataset, test_dataset=val_dataset, checkpoint_dir=checkpoint_dir), # i want to path num_epochs and the datasets
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=ray_scheduler,
        progress_reporter=reporter,
        checkpoint_at_end=False)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))
    
    best_trained_model = LSTM1(num_classes, input_size, best_trial.config['hidden_size'], best_trial.config['num_layers'], seq_len)
    saveModel(best_trained_model, 'tests/models/test_lstm_exp_raytune.pt')

if __name__ == "__main__":
    main()
