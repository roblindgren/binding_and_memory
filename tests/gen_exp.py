import numpy as np
import random
from math import floor


class expBuilderTest:
    '''
    This class is used to create Continuous Recognition Memory experiments.
    '''

    def __init__(self, img_size = 10, n_imgs = (2**10)-1, exp_size = (2**10)-1, seq_len = 2, n_initial_view = 1, n_back = 1, p = 0.5, test_split = 0.3):
        '''
        Args:
            img_size: The size of individual "image" elements. int.
            n_imgs: The total number of "images" to create for the image stack. int.
            exp_size: The length of the output arrays X and y, i.e., the number of images
                that the learner will see over the course of the experiment.
            n_initial_view: The number of images that should be shown before any are 
                repeated.
            n_back: The number of time steps back from which we will pull a previously seen image
                for "seen" images in the experiment. i.e., if n_back=1, every "seen" image will be the same
                as the image shown 1 time step before.
            p: The probability of an image in the sequence being seen (as opposed to unseen).
            test_split: The proportion of experiment images to be allocated to the validation set. 
                The remainer will go in the training set. For exp_size == 1000 and test_split == 0.3,
                700 images will be in training set and 300 in test set.
        '''
        self.img_size = img_size
        self.n_imgs = n_imgs
        self.exp_size = exp_size
        self.seq_len = seq_len
        if n_initial_view < n_back:
            self.n_initial_view = n_back+1
        else:
            self.n_initial_view = n_initial_view
        self.n_back = n_back
        self.p = p
        self.test_split = test_split

    def build_stack(self):

        # Ensure that we can create n_imgs unique binary arrays at size img_size 
        max_int = 2**self.img_size - 1 
        if max_int < self.n_imgs:
            raise Exception("img_size is too small given number of n_imgs")

        # Create a list of random integers
        source_ints = random.sample(range(self.n_imgs), self.n_imgs)
        source = []

        # Calculate max_length of binary arrays so that we can do zero-padding
        max_length = len(bin(max_int)) - 2

        # Create list of binary arrays with zero padding
        for n in source_ints:
            n_arr = [int(d) for d in bin(n)[2:]]    # Create binary array from integer
            z = [0] * (max_length - len(n_arr))     # Create leading zeros
            out = z + n_arr                         # Add leading zeros to binary array
            source.append(out)

        # Verify there are no duplicates
        for x in source:
            count = source.count(x) 
            if count > 1:
                raise Exception("There are duplicate entries in image stack")

        return source

    def build_exp(self):
        '''
        Returns:
            X: A list of lists of size img_size, n_imgs containing the experiment "images". 
            y: A list of integers randomly chosen from the set [0, 1] that server as target
                labels. 0 means the image is novel and 1 means the image has been shown earlier 
                in the experiment.
        '''
        # Make IMAGE STACK (img_source)
        # Create a list of lists, where each sublist is a unique combination of
        # zeros and ones.    
        img_source = self.build_stack()

        # Make EXPERIMENT TARGETS (y)
        # Create a random sequence of zeros and ones, also in a list
        y = np.random.choice([0,1], size = self.exp_size, p = [1-self.p, self.p]).tolist()
        bound_idx = floor(self.exp_size*(1-self.test_split))
        y_train = y[:bound_idx]
        y_test = y[bound_idx:]
        y_train[0:self.n_initial_view] = [0]*self.n_initial_view # ensure first few images are "unseen" or "new"
        y_test[0:self.n_initial_view] = [0]*self.n_initial_view # ensure first few images are "unseen" or "new"

        # Make IMAGE LIST for EXPERIMENT (X)
        # Create a new empty list the same size as the random sequence.
        # Iterate over the random sequence: for each 0, pop an element from the  
        # image stack and place it in the corresponding image list entry.
        X_train = [0] * len(y_train)
        for idx in range(len(y_train)):
            if y_train[idx] == 0:
                # get from img_source
                X_train[idx] = img_source.pop()
            else:
                # get from previously seen images
                #choice_idx = np.random.choice(idx)
                choice_idx = idx - self.n_back
                X_train[idx] = X_train[choice_idx]

        # Break X_train into sequences
        X_train_seq = []
        y_train_seq = []
        for idx in range(len(y_train)-self.seq_len+1):            
            X_train_seq.append(X_train[idx:idx+self.seq_len])
            y_train_seq.append(y_train[idx:idx+self.seq_len])
        
        X_test = [0] * len(y_test)
        for idx in range(len(y_test)):
            if y_test[idx] == 0:
                # get from img_source
                X_test[idx] = img_source.pop()
            else:
                # get from previously seen images
                #choice_idx = np.random.choice(idx)
                choice_idx = idx - self.n_back
                X_test[idx] = X_test[choice_idx]
        
        # Break X_test into sequences
        X_test_seq = []
        y_test_seq = []
        for idx in range(len(y_test)-self.seq_len+1):            
            X_test_seq.append(X_test[idx:idx+self.seq_len])
            y_test_seq.append(y_test[idx:idx+self.seq_len])

        return X_train_seq, y_train_seq, X_test_seq, y_test_seq
