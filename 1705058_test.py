import pickle
import os
from train_1705058 import *

if os.name == 'nt':
    DATA_DIR = os.path.join(os.getcwd(), 'dataset\\')
    MODEL_SAVE_DIR = os.path.join(os.getcwd(), 'model\\')
else:
    DATA_DIR = os.path.join(os.getcwd(), 'dataset/')
    MODEL_SAVE_DIR = os.path.join(os.getcwd(), 'model/')
    
# Load data
with open(MODEL_SAVE_DIR + '1705058.pickle', 'rb') as f:
    data = pickle.load(f)
    
#print(data)
        
for key in data:
    for key2 in key:
        if key2 == 'learning_rate':
            current_learning_rate = key[key2]
            
print("Learning rate: " + str(current_learning_rate))
    
# Reconstruction of the model
model = CNN()

model.add_layer(Convolution(1, 3, 1, 1, current_learning_rate, 'conv1'))
model.add_layer(ReLU(current_learning_rate, 'relu1'))
model.add_layer(MaxPool(2, 1, current_learning_rate, 'maxpool1'))
model.add_layer(Flattening(current_learning_rate, 'flatten1'))
model.add_layer(Dense(576, current_learning_rate, 'dense1'))
model.add_layer(ReLU(current_learning_rate, 'relu2'))
model.add_layer(Dense(10, current_learning_rate, 'dense2'))
model.add_layer(Softmax(current_learning_rate, 'softmax'))

for layer in model.get_layers():
    if layer.has_weights() == True:
        for key in data:
            for key2 in key:
                if key2 == layer.get_name():
                    layer.load_weights(key[key2])

print("Model loaded")

load_test_data()
test_image_data, _ = load_train_data_label(test_path_all, None)
# # Reshape into batch, channel, height, width
test_image_data = test_image_data.reshape(test_image_data.shape[0], 1, image_dimension, image_dimension).astype('float32')
test_image_data = test_image_data / 255
print("Test data loaded")

# Predict
predictions = model.predict(test_image_data)

# Output to csv
file = open('1705058_prediction.csv', 'w')

file.write('Filename,Digit\n')
for i in range(len(predictions)):
    filename = test_path_all[i].split('/')[-1]
    prediction = np.argmax(predictions[i])
    file.write(filename + ',' + str(prediction) + '\n')
    
print("Output to csv")
print("This is updated. Check if git works")
print("Hello from branchC")