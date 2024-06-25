import tensorflow as tf

# Import TensorFlow Datasets
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# Helper libraries
import math
import numpy as np
import keras

# Supervised training of fashion mnist
dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

# Class names
class_names = metadata.features['label'].names
print(f"Class names: {class_names}")

# Count of datasets
num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples

# Normalize - 255 is the max and it just divides the values by 255.
def normalize(images, labels):
  images = tf.cast(images, tf.float32)
  images /= 255
  return images, labels

# The map function applies the normalize function to each element in the train
# and test datasets
train_dataset =  train_dataset.map(normalize)
test_dataset  =  test_dataset.map(normalize)

# The first time you use the dataset, the images will be loaded from disk
# Caching will keep them in memory, making training faster
train_dataset =  train_dataset.cache()
test_dataset  =  test_dataset.cache()

print(f"Number of training examples: {num_train_examples}")
print(f"Number of test examples:     {num_test_examples}")

model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)), # Shape layer
    keras.layers.Flatten(),         # Flatten the dimensions
    keras.layers.Dense(128, activation=tf.nn.relu), # Hidden layer with ReLu
    keras.layers.Dense(len(class_names), activation=tf.nn.softmax) # Output layer with softmax for classification
])

# Regression - Loss Fn: Mean Square Error
# Classification - Loss Fn: Sparse Categorical Crossentropy
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy']) # Evaluate accuracy

BATCH_SIZE = 32 # Apparantly Batch size of 32 works well for all classification problems

# shuffles randomly
# repeats
# Batches of 32 training sets for each update of model variables
train_dataset = train_dataset.cache().repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.cache().batch(BATCH_SIZE)

# Fit the model
model.fit(train_dataset, epochs=5, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))

# Evaluate accuracy
test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples/32))

# Got 0.8791 for 128 dense ReLu. For some reason, 128-dense ReLU is better than 256 or 64 size
# Also TensorFlow 2.16 gives 10% better accuracy than TensorFlow 2.0
print('Accuracy on test dataset:', test_accuracy)

# Make predictions
for test_images, test_labels in test_dataset.take(1):
  test_images = test_images.numpy()
  test_labels = test_labels.numpy()
  predictions = model.predict(test_images)

#print(predictions)

# Prediction for the first image
# Use argmax to print the largest index
print(predictions[0], np.argmax(predictions[0]))

correct_prediction = 0
for i in range(0, 32):
  print(f"Prediction: {np.argmax(predictions[i])}, Actual: {test_labels[i]}")
  if np.argmax(predictions[i]) == test_labels[i]:
    correct_prediction = correct_prediction + 1
    
print(f"Correct predictions: {correct_prediction} / 32")
