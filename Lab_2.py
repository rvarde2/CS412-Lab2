import torch
import sklearn, sklearn.datasets, sklearn.model_selection
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
from matplotlib import pyplot as plt
import csv
import pandas as pd


# ======================================================================================================
# CLASSES
# ======================================================================================================

class LinearPotentials(torch.nn.Module):
    def __init__(self, input_dim, output_dim, random_weights_init = True):
        super(LinearPotentials, self).__init__()
        self.num_features = input_dim
        self.num_classes = output_dim
        # This networks's layers
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.history = {'train_loss':[], 'test_loss': [], 'accuracy':[]}
        if random_weights_init == True:
            self.linear.weight.data.uniform_(0.0, 1.0)
            self.linear.bias.data.fill_(0)
            
    def forward(self, x):
        outputs = self.linear(x)
        return outputs
    
    def plot_learning_curves(self, title=''):
        title = 'Learning Curves Plot' if title == '' else title
        fig = plt.figure(figsize=(14, 4))
        iters = np.arange(0, len(self.history['train_loss']))
        plt.plot(iters, self.history['train_loss'], linestyle='dashed',  label = 'Train Loss')
        plt.plot(iters, self.history['test_loss'],  linestyle='-',  label = 'Test Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Log Loss')
        plt.legend()
        plt.title(title)
        #plot(iters, self.history['train_loss'])
        plt.show()
        return fig
    
    def compute_error_stats(self):
        ''' DESCRIPTION: This class function computes the mean and std values from loggged history
                         Make sure all  test_accuracy, train_loss, test_loss are of equal size.
                         NOTE: This is for the surrent model not a collection of models.
        '''
        mean, std = 0,0

        mean_test_accuracy = np.mean(self.history['accuracy'])
        mean_train_loss = np.mean(self.history['train_loss'])
        mean_test_loss = np.mean(self.history['test_loss'])
        std_test_accuracy = np.std(self.history['accuracy'])
        print("Mean Accuracy: {}, std: {}".format(mean_test_accuracy, std))
        return mean_test_accuracy, std

# -----------------------------------------------------------------------------------------------  

class DATASET():
    ''' DESCRIPTION: Class implementing a dataset object. There may be some corner case issues.
    '''
    def __init__(self, data, normalizeData = False):
        self.data = data
        self.size = data.size()
        self.idxs = {} # holds all the indexes of the samples yet to be put to the eval set
        self.numOfSamples = self.size[0]
        if normalizeData:
            self._normalize_Data()
    # ---|
    
    def _create_eval_idx(self):
        idx = np.arange(self.numOfSamples)
        np.random.shuffle(idx)
        return idx
    
    # ---|
    
    def create_k_fold_set(self, k, resetIdx = False):
        ''' This class method creates a k-fold program for a given k. The class can maintain k-fold schedule for several k's!
            It returns 3 tensorrs or nd arrays with the train, valdiation and eval sets. WHen resetIdx = True the eval idx for that specific
            input k is remade.
        '''
        if (k not in self.idxs.keys()) or (resetIdx == True):
            k_idx = self._create_eval_idx()
            self.idxs[k] = k_idx
            
        evalSize = int(self.numOfSamples * k)
        evalIdxs = self.idxs[k][:evalSize]
        evalSet = self.data[evalIdxs]
        self.idxs[k] = self.idxs[k][evalSize:]
        # Get Eval and Train
        evalAndTrain = np.delete(self.data, evalIdxs, axis =0)
        trainSet = evalAndTrain[:int(evalAndTrain.shape[0]*0.9)]
        valSet = evalAndTrain[int(evalAndTrain.shape[0]*0.9):]
        if evalSet.shape[0] == 0: print("---------\nEvaluation set (K: {}) is empty! Run out of unseen Samples!\n---------".format(k))
            
        return trainSet, valSet, evalSet
    
    # ---|
    
    def _normalize_Data(self):
        print("Implement Data Normalization")
        
    def substitute_last_col(self, inCol):
        print(self.data.shape, inCol.shape)
        self.data[:,-1] = inCol
# ======================================================================================================
# HELPER FUNCTIONS
# ======================================================================================================

def load_data(retType = 'tensor', printData = False):
    ''' DESCRIPTION : THis function handles data loading.
    
        ARGUMENTS: retType (string): ['tensor', 'ndArray']-> selector for the return type of this function.
                                    
        RETURNS: data (ndArray or tensor)
    '''
    data = []
    
    # uSE PANDAS TO READ DATA
    df = pd.read_csv ('insurance.csv')
    
    # convert all binary variables to 1,0
    df['sex'] = df['sex'].map(lambda sex: 1 if sex =='male' else 0)
    df['smoker'] = df['smoker'].map(lambda smoker: 1 if smoker =='yes' else 0)
    c = df['charges']
    # Remove charges so we can attach it to the end of the final frame, for convienience
    df = df.drop('charges', axis = 'columns') 
    # Convert categorical variable to 1 hot encoding
    one_hot_regions = pd.get_dummies(df.region, prefix='region')
    # Remove regions columns which is strings
    df = df.drop('region', axis = 'columns') 
    # Form new dataframe
    df = pd.concat([df, one_hot_regions, c], axis = 'columns')
    
    if printData:
        print(df)
    # Convert frame to numpy
    data = df.to_numpy()
    # If needed convert frame to Pytorch Tensor.
    if retType == 'tensor':
        data = numpy_to_tensor(data)
    return data

# -----------------------------------------------------------------------------------------------

def numpy_to_tensor(data, dType=np.float32):   
    return torch.from_numpy(data.astype(dType))

# -----------------------------------------------------------------------------------------------

def plot_all_in_one(histories, saveFile = '', save=True):
    ''' DESCRIPTION: This function should accept a list of histories, labels and a title and plot all curves in one figure
                     TODO.
    '''
    ret = []
# -----------------------------------------------------------------------------------------------

def evaluate_model(model, data, optimizer, lr = 0.1, criterion = torch.nn.CrossEntropyLoss(), 
                   number_of_epochs = 10000, print_interval = 100, debug= False, print_plot=True):
    ''' DESCRIPTION: This function should facilitate training of a given PyTorch model on a given dataset. It performs Gradient Descent
                     To iterativaly update the weights of the learning, attempting to minimize training loss, at each epoch.
                     The function will also log training progress in the model's history varaible (which is a python dicitionary)
                     and plot the learning curves at the end of the training process.
                     
        ARGUMENTS: model (nn module): Learner model. nn module type
                   data (dictionary): The train/test data provided a dictionary 
                                      data = {'x_train':torch.tensor, 'x_test': torch.tensor, 'y_train':torch.tensor, 'y_test': torch.tensor}  
                   optimizer(nn optim): Chose optimizer, i.e adam, SGD etc
    
                   criterion (nn loss): NN loss function. i.e CrossEntropyLoss.
                   number_of_epochs (int): How many epochs the model should be evaluated for.
                   print_interval (int):   Per how many iters should the script print out info.
                   print_plot (boolean):    Select whether to print the learning curves.
        RETURNS: train_log_loss (float)
                 test_log_loss (float)
                 test_accuracy (float)
    '''      
    # Handle Inputs
    x_train = data['x_train']
    x_test = data['x_test']
    y_train = data['y_train']
    y_test = data['y_test']
    # ---|
    

    for epoch in range(number_of_epochs): 
        y_prediction=model(x_train)          # make predictions
        loss=criterion(y_prediction,y_train) # calculate losses
        model.history['train_loss'].append(loss.item())
        loss.backward()                      # obtain gradients
        optimizer.step()                     # update parameters
        optimizer.zero_grad()                # reset gradients
    
        
        y_prob = torch.softmax(model(x_test), 1)
        y_pred = torch.argmax(y_prob, axis=1)
        train_log_loss = criterion(model(x_train), y_train).detach().numpy()
        test_log_loss = criterion(model(x_test), y_test).detach().numpy()
        test_accuracy = (sum(y_pred==y_test)/y_test.shape[0]).detach().numpy()
        
        model.history['test_loss'].append(test_log_loss.item())
        model.history['accuracy'].append(test_accuracy)
        if (epoch+1)%print_interval == 0:
            print('Epoch:', epoch+1,',loss=',loss.item())

    # Print model parameters
    if debug == True:
        for param in model.named_parameters():
            print("Param = ",param)     
    
    print("Train Log Loss = ", train_log_loss)
    print("Test Log Loss  = ", test_log_loss)
    print("Test Accuracy  = ", test_accuracy) 
    if print_plot == True:
        model.plot_learning_curves()
    
    return train_log_loss, test_log_loss, test_accuracy

# ======================================================================================================
# METHODS
# ======================================================================================================

def knn(data, k = 15):
    ''' DESCRIPTION: Implement knn
    
    '''
    predictions = []
    
    return predictions

# -----------------------------------------------------------------------------------------------    

class NAIVEBAYES():
    ''' DESCRIPTION: Implemnet Naive Bayes
    
    '''
    def __init__(self, trainData):
        print("Train Naive Bayes")
        
    def predict(self, testDta):
        predictions = []
        return predicitons
    
# -----------------------------------------------------------------------------------------------    

# IMPLEMENT MLE AND MAP (according to one choice from the appendix)
def compute_priors(data):
    ''' DESCRIPTION: Compute the prior probability of each input class here.
    '''
    classPriors = []
    return classPriors

# PACKAGE OR IMPLEMENTATION of Random Forest/SVM

# ======================================================================================================
# TASK FUNCTION INTERFACES
# ======================================================================================================

def discover_classes(data, numOfClasses=3):
    ''' DESCRIPTION: THis function processes the input target data and clusters the cost values into three
                     distinct classes: Low: 0, Medium: 1, High: 2. It should return a numpy array or tensor
                     that transforms the contunous target column 'charges' into the above discrete labels.
        RETURNS classes (ndArray)-> Array that hold the class of each sample. Should be 1338x1.
    
    '''
    classes = []
    # YOUR CODE HERE
    # ??????????????
    # Use any clustering procedure seems pertinent: k means with k = numOfclasses, z-type clustering etc
    # z-type: disover the mean and std and cluster according to distance from mean. It might lead to inaccurate classes.
    # One way to combat that is to perform it twice: one on all data above median and one on all data below.
    
    
    
    # ??????????????
    
    # nd array
    return classes

# -----------------------------------------------------------------------------------------------    

def methods_evaluation(dataSet, *args, k_folds = [0.33, 0.2], **kwargs):
    ''' DESCRIPTION this function should accept a dataset as input and evaluate all the required classifiers.
    
                                                                        elective              bonus 
        RETURNS: retDict (dictionary): {'nb':[], 'mle', 'map', 'knn', 'rf':[] or 'svm': [], 'dnn':[]}
    '''
    retDict = {}
    
    for k in k_folds:
        for k_fold in range(1/k):
            print(k_fold)
            trainSet, valSet, evalSet = dataSet.create_k_fold_set(k=0.1)
            trainData, trainLabels = trainSet[:,:-1], trainSet[:,-1]
    # YOUR CODE HERE
    # ??????????????
            # Get Naive Bayes Results
            
            # GET MLE RESULTS
            
            # GET MAP RESULTS
            
            # GET KNN RESULTS
            
            # GET ELECTIVE RESULTS (RANFOM FOREST OR SVM)
            
            # GET DNN RESULTS (BONUS)
        # End of k_fold for loop
        # LOG ALL results for this k-fold strategy
        
        # PLOT PERFORMANCE for all methods for this k-fold strategy
    #???????????????
        
    return retDict


# -----------------------------------------------------------------------------------------------    
def multiclass_to_binary(mData):
    ''' DESCRIPTION: This function should accept as input the normal tensor or nd array data and transform the
                     labels to 0 or 1. You do not have to implement it if you prefer to handle Task 4 in your own
                     way, as long as you have a valid way of handling the label misclasification etc. NOTE: This
                     function could be used to handle Task 4, Part b where you retrain your methods. First transform
                     your data to new labels, retrain and then test.
                     
        RETURNS: bData (ndArray or tensor): containg the data with  trasnformed labels.    
    '''
    bData = []
    
    return bData

# -----------------------------------------------------------------------------------------------  

def positive_class_evaluation(dataSet, *args, **kwargs):
    ''' DESCRIPTION: This function should accept a dataset as input and evaluate all the required classifiers.
                     Alterations should be made so that when a sample is classified, we only consider 2 cases:
                     a) Class High is Positive (1), b) Classes Low and Medium are Negative (0). The function should return
                     the performance of all methods in terms of Precision, Recall and F-1.
    
                                                                        
        RETURNS: retDict (dictionary): {'nb': {'pretrained':{'precision': float, 'recall': float, 'f1':float}, 'retrained':{'precision': float, 'recall': float, 'f1':float}}, 
                                        'mle':{'pretrained':{'precision': float, 'recall': float, 'f1':float}, 'retrained':{'precision': float, 'recall': float, 'f1':float}},
                                        'map':{'pretrained':{'precision': float, 'recall': float, 'f1':float}, 'retrained':{'precision': float, 'recall': float, 'f1':float}},
                                        'knn':{'pretrained':{'precision': float, 'recall': float, 'f1':float}, 'retrained':{'precision': float, 'recall': float, 'f1':float}},
                            elective 1  'rf': {'pretrained':{'precision': float, 'recall': float, 'f1':float}, 'retrained':{'precision': float, 'recall': float, 'f1':float}},
                            elective 2  'svm':{'pretrained':{'precision': float, 'recall': float, 'f1':float}, 'retrained':{'precision': float, 'recall': float, 'f1':float}},
                            BONUS       'dnn':{'pretrained':{'precision': float, 'recall': float, 'f1':float}, 'retrained':{'precision': float, 'recall': float, 'f1':float}}}
    '''
    retDict = {'nb': {'pretrained':{'precision': 0., 'recall': 0., 'f1':0.}, 'retrained':{'precision': 0., 'recall': 0., 'f1':0.}}, 
               'mle':{'pretrained':{'precision': 0., 'recall': 0., 'f1':0.}, 'retrained':{'precision': 0., 'recall': 0., 'f1':0.}},
               'map':{'pretrained':{'precision': 0., 'recall': 0., 'f1':0.}, 'retrained':{'precision': 0., 'recall': 0., 'f1':0.}},
               'knn':{'pretrained':{'precision': 0., 'recall': 0., 'f1':0.}, 'retrained':{'precision': 0., 'recall': 0., 'f1':0.}},
               'rf': {'pretrained':{'precision': 0., 'recall': 0., 'f1':0.}, 'retrained':{'precision': 0., 'recall': 0., 'f1':0.}},
               'svm':{'pretrained':{'precision': 0., 'recall': 0., 'f1':0.}, 'retrained':{'precision': 0., 'recall': 0., 'f1':0.}},
               'dnn':{'pretrained':{'precision': 0., 'recall': 0., 'f1':0.}, 'retrained':{'precision': 0., 'recall': 0., 'f1':0.}}}

    
    # YOUR CODE HERE
    # ??????????????
    # Sainitize input
    # PART 1
    #-------
    # For all classifiers rerun the ones with the best parameters on the new dataset (with labels only 1 or 0)
    trainSet, valSet, evalSet = dataSet.create_k_fold_set(k, resetIdx = True)
    # Get Naive Bayes Results

    # GET MLE RESULTS

    # GET MAP RESULTS

    # GET KNN RESULTS

    # GET ELECTIVE RESULTS (RANFOM FOREST OR SVM)

    # GET DNN RESULTS (BONUS)
    # LOG ALL results for the pre trained classifiers
    # ----|
    
    
    # Part 2
    # A.
    # Retrain classifiers on the new dataset (Poisitive-Negative Labels)
    # B. Get results on the new Positive-Negative Test Set
    # Get Naive Bayes Results

    # GET MLE RESULTS

    # GET MAP RESULTS

    # GET KNN RESULTS

    # GET ELECTIVE RESULTS (RANFOM FOREST OR SVM)

    # GET DNN RESULTS (BONUS)
    # LOG ALL results for the pre trained classifiers
    
    # PLOT F-1 PERFORMANCE for all methods for this k-fold strategy
    #???????????????
        
    return retDict

# ======================================================================================================
# MAIN
# ======================================================================================================
def main():
    
    # TASK 0        
    # Load Data
    data = load_data(printData = True)
    # TASK 1
    # Discover classes
    # classes = discover_classes(...)
    # change data's charges to new classes.
    
    # TASK 2
    # Create dataset according to parameter k
    dataSet = DATASET(data)
    trainSet, valSet, evalSet = dataSet.create_k_fold_set(0.33, resetIdx = True)
    # TASK 3
    # Methods evaluation
    # retDict = methods_evaluation(...)
    
    # TASK 4 (Optional for group project members)
    # Positive class evaluation
    # retDict = positive_class_evaluation(...)

if __name__ == "__main__":
    main()
