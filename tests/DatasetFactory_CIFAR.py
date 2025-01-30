import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from ResNetExtractor import ResNetExtractor
import time
import random
import ray
import psutil

if torch.backends.mps.is_available():
    print("Hell yeah!")
else:
    print("Boooooo")
exit()

num_cpus = psutil.cpu_count()

class PatternedSequenceDataset(Dataset):
    def __init__(self, features, sequence_length):
        self.features = features
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        sequence_features = self.features[idx]
        sequence_pattern = torch.randint(2, size=(self.sequence_length - 1,))
        
        # Create the sequence of features based on the pattern
        for pattern_value in sequence_pattern:
            if pattern_value == 0:  # Choose a new random feature
                sequence_features = torch.cat((sequence_features, self.features[idx]), dim=0)
            elif pattern_value == 1:  # Repeat the last feature
                sequence_features = torch.cat((sequence_features, sequence_features[-1].unsqueeze(0)), dim=0)
        
        return sequence_features, sequence_pattern

import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet101

@ray.remote
def get_features(image, feature_extractor):
    image = image.unsqueeze(0)
    with torch.no_grad():
        return feature_extractor(image)

class DatasetFactory():
    def __init__(self, root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor(), 
                 sequence_length=10, num_sequences=1000, batch_size=None):
        self.root = root
        self.train = train
        self.download = download
        self.transform = transform
        self.cifar_dataset = torchvision.datasets.CIFAR10(root=self.root, train=self.train, download=self.download, transform=self.transform)
        self.sequence_length = sequence_length
        self.num_sequences = num_sequences
        self.batch_size = batch_size
        self.resnet_extractor = ResNetExtractor()  # Initialize the ResNet feature extractor

    def precompute_features(self, dataset):
        #features = []

        # Set the ResNetExtractor to evaluation mode
        self.resnet_extractor.eval()

        # Iterate over the dataset and extract features
        
        futures = [get_features.remote(image, self.resnet_extractor) for image, _ in dataset]
        features = ray.get(futures)
        '''
        #cnt = 0
        for image, _ in dataset:
            #print('image # ', cnt)
            # Unsqueeze to add a batch dimension
            image = image.unsqueeze(0)
            feature = self.resnet_extractor(image)
            features.append(feature)
            #cnt += 1
        '''

        # Restore the ResNetExtractor to its original mode
        self.resnet_extractor.train()

        return features

    def getData(self):
        # Calculate the maximum size for subsets
        max_subset_size = self.sequence_length * self.num_sequences

        # Limit the dataset size for subsets if needed
        if len(self.cifar_dataset) > max_subset_size:
            # Generate random indices for the subset
            random_indices = random.sample(range(len(self.cifar_dataset)), max_subset_size)

            # Create a subset using the randomly chosen indices
            self.cifar_dataset = torch.utils.data.Subset(self.cifar_dataset, random_indices)

        # Split the dataset into training, testing, and holdout subsets
        total_samples = len(self.cifar_dataset)
        train_ratio = 0.6  # Example: 60% for training
        test_ratio = 0.2   # Example: 20% for testing
        train_size = int(total_samples * train_ratio)
        test_size = int(total_samples * test_ratio)
        holdout_size = total_samples - train_size - test_size

        train_subset, test_subset, holdout_subset = torch.utils.data.random_split(
            self.cifar_dataset, [train_size, test_size, holdout_size]
        )

        # Precompute features for each subset
        print("Precompute train_features")
        train_features = self.precompute_features(train_subset)
        print("Precompute test_features")
        test_features = self.precompute_features(test_subset)
        print("Precompute holdout_features")
        holdout_features = self.precompute_features(holdout_subset)

        # Create custom datasets for each subset using the precomputed features
        train_dataset = PatternedSequenceDataset(train_features, self.sequence_length)
        test_dataset = PatternedSequenceDataset(test_features, self.sequence_length)
        holdout_dataset = PatternedSequenceDataset(holdout_features, self.sequence_length)

        # Create data loaders for each subset
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        holdout_loader = DataLoader(holdout_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader, holdout_loader


# Time initialization
if __name__ == "__main__":
    start = time.time()
    factory = DatasetFactory()
    train, test, holdout = factory.getData()
    end = time.time()
    print('Time to initialize DatasetFactory when exracting features for all CIFAR: ', end-start)