# Mine
from model_LSTM import LSTMTrainer
from DatasetFactory_CIFAR import DatasetFactory
from binary_acc import binary_acc

# PyTorch
import torch
import torch
import torch.nn as nn
from torchvision import transforms

# Raytune
import ray
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

# Utilities
import psutil
import os
from functools import partial

def train_raytune(config, model, seq_len, train_dataset_id, test_dataset_id, num_epochs=10, checkpoint_dir=None):
    # Instantiate the LSTM
    model = LSTMTrainer(config['input_size'], config['hidden_size'], config['num_layers'], seq_len-1) #our lstm class 

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    #model.to(device)

    loss_fn = nn.BCEWithLogitsLoss() # THIS IS IMPORTANT!!! What I used on the integer version that actually learned, let's see if it helps here.
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['gamma'], verbose=True)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    train_loader = ray.get(train_dataset_id)
    test_loader = ray.get(test_dataset_id)
    #print('len(train_loader):', len(train_loader))
    #print('len(test_loader):', len(test_loader))

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0 
        running_test_loss = 0.0
        running_train_acc = 0.0 
        running_test_acc = 0.0 
        train_n = 0
        test_n = 0
        for train_batch, train_labels in train_loader:
            #print('train_batch.size: ', train_batch.size()) # this becomes x in forward()
            # Reshape train_batch
            train_batch = train_batch.view(1, train_batch.shape[0], -1)
            #print('train_batch.size after view: ', train_batch.size())

            # zero the parameter gradient
            optimizer.zero_grad() #calculate the gradient, manually setting to 0

            # Forward, backward, and optimize
            train_out = model.forward(train_batch) #forward pass
            # Use train_out[0] b/c we aren't batching, so we need grab the output for 1 batch
            train_out = train_out[0]
            #print('train_out: ', train_out)
            #print('train_labels: ', train_labels)
            train_loss = loss_fn(train_out, train_labels.float()) # fix all these goofy 1:s. train_label and test_label should be seq_len-1 length.
            train_loss.backward() # backprop loss
            optimizer.step() #improve from loss, i.e backprop

            # calc train stats
            running_train_loss += train_loss.item()
            #print('train_out.size(0):', train_out.size(0))
            train_n += train_out.size(0) 
            #print('train_out.shape: ', train_out.shape)
            #print('train_labels.shape: ', train_labels.shape)
            train_acc = binary_acc(train_out, train_labels)
            running_train_acc += train_acc.item()*train_batch.size(0)

        # Calculate test loss and accuracy
        with torch.no_grad():
            model.eval()
            for test_batch, test_labels in test_loader:
                test_batch, test_labels  = test_batch.to(device), test_labels.to(device)

                #print('test_batch.size: ', test_batch.size()) # this becomes x in forward()
                # Reshape test_batch
                test_batch = test_batch.view(1, test_batch.shape[0], -1)
                #print('test_batch.size after view: ', test_batch.size())

                test_out = model.forward(test_batch)
                # Use test_out[0] b/c we aren't batching, so we need grab the output for 1 batch
                test_out = test_out[0]
                #print('test_out: ', test_out)
                #print('test_labels: ', test_labels)
                test_loss = loss_fn(test_out, test_labels.float())
                running_test_loss += test_loss.item()
                test_n += test_out.size(0) 
                test_acc = binary_acc(test_out, test_labels)
                running_test_acc += test_acc.item()*test_batch.size(0)
        scheduler.step()
        
        # Calculate performance states for epoch
        #print('train_n:', train_n)
        #print('test_n:', test_n)
        train_loss_epoch = running_train_loss/len(train_loader) 
        test_loss_epoch = running_test_loss/len(test_loader)
        train_acc_epoch = (running_train_acc)/train_n
        test_acc_epoch = (running_test_acc)/test_n
        #print('train_acc_epoch: ', train_acc_epoch)
        #print('test_acc_epoch: ', test_acc_epoch)

        if checkpoint_dir:
            with ray.tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict()), path)

        ray.tune.report(train_loss=train_loss_epoch, train_accuracy=train_acc_epoch, test_loss=test_loss_epoch, test_accuracy=test_acc_epoch)

# Function to save the model 
def saveModel(model, path): 
    torch.save(model.state_dict(), path)

def main():    
    # Set directories
    data_dir = os.path.abspath("./tests/raytune_data/")
    #checkpoint_dir = os.path.abspath("./tests/raytune_checkpoint/")
    checkpoint_dir = None
    print('checkpoint_dir is:', checkpoint_dir)

    # Experiment parameter
    seq_len = 2
    exp_size = 2000
    
    # Tune parameters
    num_samples=1 # Number of times to try the same hyperparameters
    max_num_epochs=100

    # Set the num of cpus and fraction of cpu memory to use
    mem_frac=0.95
    num_cpus=psutil.cpu_count()
    mem_per_cpu=psutil.virtual_memory().total / num_cpus
    mem_per_trial=int(mem_per_cpu*mem_frac)
    gpus_per_trial=0
    cpus_per_trial=num_cpus-1
    

    #os.environ['TEMPDIR'] = "/Volumes/Rob's Passport 2/Kording Lab/"
    #ray.init(num_cpus=num_cpus-2, _plasma_directory="/Volumes/Rob's Passport 2/Kording Lab/")
    ray.init()
    print(ray.cluster_resources())
    
    config = {
        #"hidden_size": ray.tune.grid_search([100, 200, 300, 400]),
        "hidden_size": 100,
        #"num_layers": ray.tune.grid_search([2, 3, 4]),
        "num_layers": 1,
        #"lr": ray.tune.grid_search([1e-6, 1e-5, 1e-4, 1e-3]),
        "lr": 1e-6,
        #"batch_size": ray.tune.grid_search([32, 64, 128]),
        "batch_size": None, #32
        #"gamma": ray.tune.grid_search([0.9, 0.99, 0.999])
        "gamma": 0.999
    }
    
    # New data generator
    # Define data transform
    data_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
    dataFactory = DatasetFactory(transform=data_transform, sequence_length=seq_len, num_sequences=exp_size, batch_size=config['batch_size'])
    train_dataset, val_dataset, test_dataset = dataFactory.getData()
    train_dataset_id = ray.put(train_dataset)
    val_dataset_id = ray.put(val_dataset)
    test_dataset_id = ray.put(test_dataset)

    # Get a batch of data (you can get the first batch)
    for features, _ in train_dataset:

        # Get the input_size from the shape of the features
        print('features.shape: ', features.shape)
        input_size = features.shape[1]

        # Print the input_size
        print("Input size for LSTM: ", input_size)
        config['input_size'] = input_size
        
        # Break after processing the first batch
        break
    
    
    
    '''
    print('train_dataset type:', type(train_dataset))
    print('train_dataset len:', len(train_dataset))
    print('train_dataset[0][0] type:', type(train_dataset[0][0]))
    print('train_dataset[0][0] len:', len(train_dataset[0][0]))
    print('train_dataset[0][0][0] shape:', train_dataset[0][0][0].shape)
    print('train_dataset first item:', train_dataset.__getitem__(0)) 
    '''
    
    ray_scheduler = ASHAScheduler(
        metric="train_loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=2,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["hidden_size", "num_layers", "lr", "batch_size", "gamma"],
        metric_columns=["train_loss", "train_accuracy", "training_iteration"])
    result = ray.tune.run(
        partial(train_raytune, model=LSTMTrainer, seq_len=seq_len, num_epochs=max_num_epochs, train_dataset_id=train_dataset_id, test_dataset_id=val_dataset_id, checkpoint_dir=checkpoint_dir),
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial, "memory": mem_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=ray_scheduler,
        progress_reporter=reporter,
        checkpoint_at_end=False, 
        local_dir = "./ray_results")
        
    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final test loss: {}".format(
        best_trial.last_result["test_loss"]))
    print("Best trial final test accuracy: {}".format(
        best_trial.last_result["test_accuracy"]))
    
    #input_size, hidden_size, num_layers, output_size
    best_trained_model = LSTMTrainer(best_trial.config['input_size'], best_trial.config['hidden_size'], 
                                     best_trial.config['num_layers'], seq_len-1)
    saveModel(best_trained_model, 'tests/models/test_lstm_exp_raytune.pt')

if __name__ == "__main__":
    main()
