import os
import cv2
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.metrics import f1_score, accuracy_score
# try:
#     import cupy as np
# except:
#     import numpy as np

####################################### Variables #######################################
train_validation_ratio = 0.8
batch_size = 256
epochs = 5
epsilon = 1e-8

learning_rates = [0.01, 0.001, 0.0001, 0.00001]

if os.name == 'nt':
    DATA_DIR = os.path.join(os.getcwd(), 'dataset\\')
    MODEL_SAVE_DIR = os.path.join(os.getcwd(), 'model\\')
else:
    DATA_DIR = os.path.join(os.getcwd(), 'dataset/')
    MODEL_SAVE_DIR = os.path.join(os.getcwd(), 'model/')

train_path_all = []
train_label_csv_path = []
test_path_all = []
test_label_csv_path = []

train_folders = ['b']
test_folders = ['d']

image_dimension = 28

####################################### Building Blocks #######################################
class ReLU:
    def __init__(self, learning_rate):
        # mask: boolean, x <= 0 => True, otherwise => False
        self.mask = None
        self.not_mask = None
        self.input_matrix = None
        self.is_learnable = False
        
        self.learning_rate = learning_rate
        
        self.Name = "ReLU"
        
    def has_weights(self):
        return self.is_learnable
    
    def get_name(self):
        return self.Name
    
    def set_name(self, name):
        self.Name = name

    # Forward: if x <= 0, then output = 0, otherwise output = x
    def forward(self, x):
        self.input_matrix = x
        self.mask = (x <= 0)

        # Copy the given array to output variable
        out = x.copy()
        # Apply mask to output variable
        out[self.mask] = 0
        
        self.set_name('ReLU' + str(x.shape) + '->' + str(out.shape))
        
        return out

    # Backward: if x <= 0, then output = 0, otherwise output = 1 (Derivative of ReLU)
    def backward(self, back):
        dout = self.input_matrix.copy()
        self.mask = (back <= 0)
        self.not_mask = (back > 0)
        # Apply mask to backpropagated error
        dout[self.mask] = 0
        dout[self.not_mask] = 1
        
        self.set_name('ReLU ' + str(back.shape) + '->' + str(dout.shape))
        
        return back * dout
    
class MaxPool:
    def __init__(self, filter_dimension, stride, learning_rate):
        self.filter_dimension = filter_dimension
        self.stride = stride
        self.is_learnable = False
        self.input_matrix = None
        
        self.learning_rate = learning_rate
        
        self.Name = "MaxPool (" + str(filter_dimension) + ", " + str(stride) + ")"
        
    def has_weights(self):
        return self.is_learnable
    
    def get_name(self):
        return self.Name
    
    def set_name(self, name):
        self.Name = name
    
    # Forward: Find the maximum value in each region
    def forward(self, input_matrix):
        # self.input_matrix = input_matrix
        # batch_size, input_channels, input_height, input_width = input_matrix.shape
        # filter_height, filter_width = self.filter_dimension, self.filter_dimension
        
        # output_height = int(1 + (input_height - filter_height) / self.stride)
        # output_width = int(1 + (input_width - filter_width) / self.stride)
        # output_channels = input_channels
        
        # # Create shape
        # shape = (batch_size, output_channels, output_height, output_width, filter_height, filter_width)
        
        # # Number of bytes to jump to get the next element
        # stride = input_matrix.strides
        # # Update stride along the height and width dimension
        # stride = (stride[0], stride[1], self.stride * stride[2], self.stride * stride[3], stride[2], stride[3])
        # # Create window according to shape
        # windows = np.lib.stride_tricks.as_strided(input_matrix, shape=shape, strides=stride)
        
        # # Max pooling along the height and width dimension
        # A = np.max(windows, axis=(4, 5))
        
        # self.set_name('MaxPool(' + str(self.filter_dimension) + ', ' + str(self.stride) + ')' + str(input_matrix.shape) + '->' + str(A.shape))
        
        # return A
        self.input_matrix = input_matrix
        batch_size, input_channels, input_height, input_width = input_matrix.shape
        filter_height, filter_width = self.filter_dimension, self.filter_dimension
        
        output_height = int(1 + (input_height - filter_height) / self.stride)
        output_width = int(1 + (input_width - filter_width) / self.stride)
        output_channels = input_channels
        
        output_matrix = np.zeros((batch_size, output_channels, output_height, output_width))
        
        for batch in range(batch_size):
            for height in range(output_height):
                for width in range(output_width):
                    vertical_start = height * self.stride
                    vertical_end = vertical_start + self.filter_dimension
                    horizontal_start = width * self.stride
                    horizontal_end = horizontal_start + self.filter_dimension
                    
                    for channel in range(input_channels):
                        x = input_matrix[batch, channel, vertical_start:vertical_end, horizontal_start:horizontal_end]
                        output_matrix[batch, channel, height, width] = np.max(x)
        
        self.set_name('MaxPool(' + str(self.filter_dimension) + ', ' + str(self.stride) + ')' + str(input_matrix.shape) + '->' + str(output_matrix.shape))
        
        return output_matrix

    # Backward: Find the index of the maximum value in each region and set the error to that index
    def backward(self, error_matrix):
        error_batch_size, error_channels, error_height, error_width = error_matrix.shape
        
        previous_input_matrix = self.input_matrix
        dout = np.zeros_like(previous_input_matrix)
        
        for batch in range(previous_input_matrix.shape[0]):
            for height in range(error_height):
                for width in range(error_width):
                    vertical_start = height * self.stride
                    vertical_end = vertical_start + self.filter_dimension
                    horizontal_start = width * self.stride
                    horizontal_end = horizontal_start + self.filter_dimension
                    
                    for channel in range(error_channels):
                        # Find the index of the maximum value in each region
                        x = previous_input_matrix[batch, channel, vertical_start:vertical_end, horizontal_start:horizontal_end]
                        mask = (x == np.max(x))
                        dout[batch, channel, vertical_start:vertical_end, horizontal_start:horizontal_end] += mask * error_matrix[batch, channel, height, width]
        
        self.set_name('MaxPool(' + str(self.filter_dimension) + ', ' + str(self.stride) + ')' + str(error_matrix.shape) + '->' + str(dout.shape))
        
        return dout
    
class Flattening:
    def __init__(self, learning_rate):
        self.input_matrix = None
        self.is_learnable = False
        
        self.Name = "Flattening"
        
        self.learning_rate = learning_rate
        
    def get_name(self):
        return self.Name
    
    def set_name(self, name):
        self.Name = name
        
    def has_weights(self):
        return self.is_learnable

    # Forward: Flatten the input matrix
    def forward(self, input_matrix):
        # Store the shape of the input matrix
        self.input_matrix = input_matrix.copy()
        # Copy the input matrix to output variable
        output_matrix = input_matrix.copy()
        # Flatten the input matrix into 2D array with one column
        output_matrix = output_matrix.reshape(output_matrix.shape[0], -1)
        output_matrix = output_matrix.transpose()
        
        self.set_name('Flattening ' + str(input_matrix.shape) + '->' + str(output_matrix.shape))
        
        return output_matrix

    # Backward: Reshape the input matrix
    def backward(self, error_matrix):
        # Copy the backpropagated error matrix to output variable
        error_input_matrix = error_matrix.copy()
        # Transpose the backpropagated error matrix
        error_input_matrix = error_input_matrix.transpose()
        # Reshape the backpropagated error matrix
        error_input_matrix = error_input_matrix.reshape(self.input_matrix.shape)
        
        self.set_name('Flattening ' + str(error_matrix.shape) + '->' + str(error_input_matrix.shape))
        
        return error_input_matrix

# Must be after Flattening layer
class Dense:
    def __init__(self, output_dimension, Learning_Rate):
        # Given parameter
        self.output_dimension = output_dimension
        self.is_learnable = True
        
        # Learnable parameters
        self.input_shape = None
        self.weights = None
        self.biases = None
        self.input_vector = None
        
        self.Name = "Dense"
        
        self.learning_rate = Learning_Rate
        
    def get_name(self):
        return self.Name
    
    def set_name(self, name):
        self.Name = name
        
    def has_weights(self):
        return self.is_learnable
    
    def save_weights(self):
        dump = {
            'dense_weights': self.weights,
            'dense_biases': self.biases
        }
        return dump
        
    def he_initialization(self, previous_layer, current_layer):
        # He initialization
        # https://arxiv.org/pdf/1502.01852.pdf
        self.weights = np.random.randn(current_layer, previous_layer) * np.sqrt(2 / previous_layer)
        self.biases = np.zeros((current_layer, 1))
        
    # Forward: y = Wx + b
    def forward(self, input_vector):
        self.input_shape = input_vector.shape
        self.input_vector = input_vector
        
        # Initialize weights and biases if they are not initialized
        if self.weights is None:
            self.he_initialization(self.input_shape[0], self.output_dimension)
        
        output_vector = np.dot(self.weights, input_vector) + self.biases
        
        self.set_name('Dense(' + str(self.output_dimension) + ')' + str(input_vector.shape) + '->' + str(output_vector.shape))
        
        return output_vector
    
    # Backward: dL/dW = dL/dy * dy/dW = dL/dy * x, dL/db = dL/dy * dy/db = dL/dy, dL/dx = dL/dy * dy/dx = dL/dy * W
    def backward(self, error_vector):
        # Error with respect to the weights
        # Divide by batch size so that the result is consistent across different batch size
        error_weights = np.dot(error_vector, self.input_vector.T) / error_vector.shape[1]
        
        # Error with respect to the biases
        # Mean is used so that the result is consistent across different batch size
        error_biases = np.mean(error_vector, axis=1)
        error_biases = error_biases.reshape(error_biases.shape[0], 1)
        
        # Error with respect to the input vector
        error_input_vector = np.dot(self.weights.T, error_vector)
        
        # Update the weights and biases
        self.weights -= error_weights * self.learning_rate
        self.biases -= error_biases * self.learning_rate
        
        self.set_name('Dense(' + str(self.output_dimension) + ')' + str(error_vector.shape) + '->' + str(error_input_vector.shape))
        
        return error_input_vector

class Softmax:
    def __init__(self, learning_rate):
        self.probs = None
        self.is_learnable = False
        
        self.Name = "Softmax"
        
        self.learning_rate = learning_rate
        
    def get_name(self):
        return self.Name
    
    def set_name(self, name):
        self.Name = name
        
    def has_weights(self):
        return self.is_learnable

    # Normalize the probabilities
    def forward(self, input_vector):
        # Apply exp on each element
        exp_data = np.exp(input_vector)
        # Normalize
        exp_data = exp_data / np.sum(exp_data, axis=0)
        self.probs = exp_data
        
        self.set_name('Softmax ' + str(input_vector.shape) + '->' + str(self.probs.shape))
        
        return self.probs

    # Gradient of the error with respect to the input
    def backward(self, error_vector):
        self.set_name('Softmax ' + str(error_vector.shape) + '->' + str(error_vector.shape))
        return error_vector.copy()
    
class Convolution:
    def __init__(self, output_channels, filter_dimension, stride, padding, Learning_Rate):
        # Hyperparameters
        self.output_channels = output_channels
        self.filter_dimension = filter_dimension
        self.stride = stride
        self.padding = padding
        
        # Necessary variables
        self.filters = None
        self.biases = None
        self.is_learnable = True
        
        # Input matrix
        self.input_matrix_saved = None
        
        self.Name = "Convolution (" + str(output_channels) + ", " + str(filter_dimension) + ", " + str(stride) + ", " + str(padding) + ")"
        
        self.learning_rate = Learning_Rate
        
    def get_name(self):
        return self.Name
    
    def set_name(self, name):
        self.Name = name
        
    def has_weights(self):
        return self.is_learnable
    
    def save_weights(self):
        dump = {
            'convolution_filters': self.filters,
            'convolution_biases': self.biases
        }
        return dump
        
    def he_initialization(self, previous_layer):
        # He initialization
        # https://arxiv.org/pdf/1502.01852.pdf
        self.filters = np.random.randn(self.output_channels, previous_layer, self.filter_dimension, self.filter_dimension) * np.sqrt(2 / (previous_layer * self.filter_dimension * self.filter_dimension))
        self.biases = np.zeros(self.output_channels)
        
    def forward(self, input_matrix):
        self.input_matrix_saved = input_matrix
        batch_size, input_channels, input_height, input_width = input_matrix.shape
        
        output_height = int((input_height - self.filter_dimension + 2 * self.padding) / self.stride + 1)
        output_width = int((input_width - self.filter_dimension + 2 * self.padding) / self.stride + 1)
        
        output_matrix = np.zeros((batch_size, self.output_channels, output_height, output_width))
        
        if self.filters is None:
            self.he_initialization(input_channels)
            
        # Pad the input matrix
        padded_input_matrix = np.pad(input_matrix, ((0, ), (0, ), (self.padding, ), (self.padding, )), 'constant')
        
        # Loop through the input matrix
        for i in range(batch_size):
            sliced_input_matrix = padded_input_matrix[i]
            for j in range(output_height):
                for k in range(output_width):
                    vertical_start = j * self.stride
                    vertical_end = vertical_start + self.filter_dimension
                    horizontal_start = k * self.stride
                    horizontal_end = horizontal_start + self.filter_dimension
                    
                    for l in range(self.output_channels):
                        sliced_slice = sliced_input_matrix[:, vertical_start:vertical_end, horizontal_start:horizontal_end]
                        output_matrix[i, l, j, k] = np.sum(sliced_slice * self.filters[l]) + self.biases[l]
        
        self.set_name('Convolution(' + str(self.output_channels) + ', ' + str(self.filter_dimension) + ', ' + str(self.stride) + ', ' + str(self.padding) + ') ' + str(input_matrix.shape) + '->' + str(output_matrix.shape))
        
        return output_matrix
        
    def backward(self, error_matrix):
        batch_size, channels = self.input_matrix_saved.shape[0], self.input_matrix_saved.shape[1]
        output_height, output_width = error_matrix.shape[2], error_matrix.shape[3]
        
        dw = np.zeros(self.filters.shape)
        dout = np.zeros(self.input_matrix_saved.shape)
        db = np.zeros(self.biases.shape)
        
        padded_input_matrix = np.pad(self.input_matrix_saved, ((0, ), (0, ), (self.padding, ), (self.padding, )), 'constant')
        padded_dout = np.pad(dout, ((0, ), (0, ), (self.padding, ), (self.padding, )), 'constant')
        
        for i in range(batch_size):
            sliced_input_matrix = padded_input_matrix[i]
            sliced_dout = padded_dout[i]
            
            for j in range(output_height):
                for k in range(output_width):
                    vertical_start = j * self.stride
                    vertical_end = vertical_start + self.filter_dimension
                    horizontal_start = k * self.stride
                    horizontal_end = horizontal_start + self.filter_dimension
                    
                    for l in range(self.output_channels):
                        sliced_slice = sliced_input_matrix[:, vertical_start:vertical_end, horizontal_start:horizontal_end]
                        
                        sliced_dout[:, vertical_start:vertical_end, horizontal_start:horizontal_end] += self.filters[l] * error_matrix[i, l, j, k]
                        dw[l] += error_matrix[i, l, j, k] * sliced_slice
                        db[l] += error_matrix[i, l, j, k]
            
            dout[i] = sliced_dout[:, self.padding:-self.padding, self.padding:-self.padding]
        
        db = db / batch_size
        dw = dw / batch_size
        # dout = dout / batch_size   
        
        # Update the filters and biases
        self.filters -= self.learning_rate * dw
        self.biases -= self.learning_rate * db
        
        self.set_name('Convolution(' + str(self.output_channels) + ', ' + str(self.filter_dimension) + ', ' + str(self.stride) + ', ' + str(self.padding) + ') ' + str(error_matrix.shape) + '->' + str(dout.shape))    
        return dout
####################################### CNN Model #######################################
class CNN:
    def __init__(self) -> None:
        self.layers = []
        
    def add_layer(self, layer):
        self.layers.append(layer)
        
    def cross_entropy_loss(self, prediction, label):
        loss = -np.sum(label * np.log(prediction + epsilon)) / batch_size
        return loss
    
    def accuracy(self, prediction, label):
        prediction = np.argmax(prediction, axis=1)
        label = np.argmax(label, axis=1)
        return np.sum(prediction == label) / len(prediction)
    
    def macro_f1_score(self, prediction, label):
        prediction = np.argmax(prediction, axis=1)
        label = np.argmax(label, axis=1)
        return f1_score(label, prediction, average='macro')
    
    def print_architecture(self):
        print('CNN Architecture:')
        for layer in self.layers:
            print(layer.get_name())
        
    def forward_propagation(self, input_matrix):
        print('\nForward Propagation...')
        for layer in self.layers:
            input_matrix = layer.forward(input_matrix)
            #print(layer.get_name())
        return input_matrix
    
    def backward_propagation(self, error_matrix):
        print('Backward Propagation...')
        for layer in reversed(self.layers):
            error_matrix = layer.backward(error_matrix)
            #print(layer.get_name())
        return error_matrix
    
    def train(self, training_data, training_labels):
        for i in range(0, len(training_data), batch_size):
            batch_data = training_data[i:i + batch_size]
            batch_labels = training_labels[i:i + batch_size]
            
            # Predict on the batch
            predictions = self.forward_propagation(batch_data).T
            
            loss = self.cross_entropy_loss(predictions, batch_labels)
            print('Batch ' + str((i // batch_size) + 1) + ', Loss: ' + str(loss))
            
            loss_derivative = (predictions - batch_labels).T
            self.backward_propagation(loss_derivative)
                
    def predict(self, test_data):
        predictions = self.forward_propagation(test_data).T
        return predictions
    
    def save_model(self, learning_rate):
        if not os.path.exists(MODEL_SAVE_DIR):
            os.makedirs(MODEL_SAVE_DIR)
        
        model_name = 'model_'
        for layer in self.layers:
            model_name += str(layer.get_name()[0])
        
        # learning rate after decimal point
        model_name += str(learning_rate).split(sep='.')[1] + ".pickle"
        
        # Get the model parameters
        model_parameters = []
        for layer in self.layers:
            if layer.has_weights() == True:
                model_parameters.append(layer.save_weights())
        
        # Save the model parameters
        print('Saving model...')
        print('Model name: ' + model_name)
        file = open(MODEL_SAVE_DIR + model_name, 'wb')
        
        if file:
            pickle.dump(model_parameters, file)
            print('Model saved successfully.')
        else:
            print('Model saving failed.')

####################################### Input and Initialization #######################################        
def get_key(path):
    # seperates the key of an image from the filepath
    key=path.split(sep=os.sep)[-1]
    return key

def load_data_paths(folder_name, b_train):
    if os.name == 'nt':
        path = DATA_DIR + folder_name + '\\'
    else:
        path = DATA_DIR + folder_name + '/'
    for file in sorted(os.listdir(path)):
        if file.endswith('.png'):
            file_path = path + file
            if(b_train):
                train_path_all.append(file_path)
            else:
                test_path_all.append(file_path)
    
def load_train_data():
    for folder in train_folders:
        load_data_paths('training-' + folder, True)
        
def load_test_data():
    for folder in test_folders:
        load_data_paths('training-' + folder, False)
    
def load_train_label_csv():
    for folder in train_folders:
        csv_name = 'training-' + folder + '.csv'
        csv_path = DATA_DIR + csv_name
        train_label_csv_path.append(csv_path)
        
def load_test_label_csv():
    for folder in test_folders:
        csv_name = 'training-' + folder + '.csv'
        csv_path = DATA_DIR + csv_name
        test_label_csv_path.append(csv_path)
    
def load_train_data_label(data_path, csv_path):
    local_image_data = []
    label_data_local = []
    
    # Concatenate all label data into one DataFrame
    train_label = pd.concat([pd.read_csv(path) for path in csv_path], ignore_index=True)
    train_label = train_label.set_index('filename')
    label_data_local=[train_label.loc[get_key(path)]['digit'] for path in  data_path]
    
    for i, img_path in enumerate(data_path):
        img=cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img,(image_dimension,image_dimension),interpolation=cv2.INTER_AREA)
        gaussian_3 = cv2.GaussianBlur(img, (9,9), 10.0) #unblur
        img = cv2.addWeighted(img, 1.5, gaussian_3, -0.5, 0, img)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) #filter
        img = cv2.filter2D(img, -1, kernel)
        local_image_data.append(img) # expand image to 28x28x1 and append to the list
        if i == len(data_path)-1:
            end = '\n'
        else:
            end = '\r'
        print('processed {}/{}'.format(i+1,len(data_path)),end=end)
        
    local_image_data = np.array(local_image_data)
    return local_image_data, np.array(label_data_local)

def subsample_data(image_data, label_data, num_classes, num_samples_per_class_low, num_samples_per_class_high):
    sampled_indices = []
    
    for i in range(num_classes):
        number_of_samples = np.random.randint(num_samples_per_class_low, num_samples_per_class_high)
        indices = np.where(label_data == i)[0]
        #print('Class {}: {} samples'.format(i, len(indices)))
        sampled_indices.extend(indices[:number_of_samples])
    
    indices = np.array(sampled_indices)
    np.random.shuffle(indices)
    
    sampled_image_data = image_data[indices]
    sampled_label_data = label_data[indices]
    
    return sampled_image_data, sampled_label_data
    
####################################### Main #######################################
if __name__ == '__main__':
    load_train_data()
    load_train_label_csv()
    # load_test_data()
    # load_test_label_csv()
    
    train_image_data, train_label_data = load_train_data_label(train_path_all, train_label_csv_path)
    # Reshape into batch, channel, height, width
    train_image_data = train_image_data.reshape(train_image_data.shape[0], 1, image_dimension, image_dimension).astype('float32')
    print("Loaded training data")
    print("Training image data shape: ", train_image_data.shape)
    
    # test_image_data, test_label_data = load_train_data_label(test_path_all, test_label_csv_path)
    # # Reshape into batch, channel, height, width
    # test_image_data = test_image_data.reshape(test_image_data.shape[0], 1, image_dimension, image_dimension).astype('float32')
    # print("Loaded test data")
    # print("Test image data shape: ", test_image_data.shape)
    
    # Normalize the data
    train_image_data = train_image_data / 255
    # test_image_data = test_image_data / 255
    
    # Shuffle the indices
    indices = np.arange(train_image_data.shape[0])
    np.random.shuffle(indices)
    
    # Split the data into training and validation
    image_for_training = train_image_data[indices[:int(train_image_data.shape[0] * 0.8)]]
    label_for_training = train_label_data[indices[:int(train_image_data.shape[0] * 0.8)]]
    
    image_for_validation = train_image_data[indices[int(train_image_data.shape[0] * 0.8):]]
    label_for_validation = train_label_data[indices[int(train_image_data.shape[0] * 0.8):]]
    
    print("Split the data into training and validation")
    print("Training image data shape: ", image_for_training.shape)
    print("Validation image data shape: ", image_for_validation.shape)
    
    current_learning_rate = learning_rates[2]
    
    # Create the model
    model = CNN()
    
    # Architecture:
    model.add_layer(Convolution(1, 3, 1, 1, current_learning_rate))
    model.add_layer(ReLU(current_learning_rate))
    model.add_layer(MaxPool(2, 1, current_learning_rate))
    model.add_layer(Flattening(current_learning_rate))
    model.add_layer(Dense(576, current_learning_rate))
    model.add_layer(ReLU(current_learning_rate))
    model.add_layer(Dense(10, current_learning_rate))
    model.add_layer(Softmax(current_learning_rate))
    
    # Check the model
    model.print_architecture()
    
    # Train the model
    for epoch in tqdm(range(epochs)):
        print("Epoch: ", epoch, "/", epochs)
        
        # Subsample the data
        sampled_image_for_training, sampled_label_for_training = subsample_data(image_for_training, label_for_training, 10, 400, 700)
        sampled_image_for_validation, sampled_label_for_validation = subsample_data(image_for_validation, label_for_validation, 10, 50, 200)

        training_label = []
        validation_label = []
        
        # One hot encode the labels
        for i in range(len(sampled_label_for_training)):
            one_hot = np.zeros(10)
            one_hot[sampled_label_for_training[i]] = 1
            training_label.append(one_hot)
            
        for i in range(len(sampled_label_for_validation)):
            one_hot = [0] * 10
            one_hot[sampled_label_for_validation[i]] = 1
            validation_label.append(one_hot)
            
        sampled_label_for_training = np.array(training_label)
        sampled_label_for_validation = np.array(validation_label)
        
        # label_for_training = np.eye(10)[np.array(label_for_training).reshape(-1)]
        # label_for_validation = np.eye(10)[np.array(label_for_validation).reshape(-1)]
        
        model.train(sampled_image_for_training, sampled_label_for_training)
        
        predicted_labels = model.predict(sampled_image_for_validation)
        
        # Print the predicted labels
        print(predicted_labels)
        
        # Calculate the cross entropy, accuracy and macro F1 score loss for validation data
        loss = model.cross_entropy_loss(predicted_labels, sampled_label_for_validation)
        accuracy = model.accuracy(predicted_labels, sampled_label_for_validation)
        f1 = model.macro_f1_score(predicted_labels, sampled_label_for_validation)
        
        print("Validation Loss: ", loss)
        print("Validation Accuracy: ", accuracy)
        print("Validation Macro F1 Score: ", f1)
    
    # Save the model
    model.save_model(current_learning_rate)