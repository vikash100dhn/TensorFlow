import tensorflow as tf;
import os, sys;
import skimage;
from skimage.color import rgb2gray
import matplotlib.pyplot as plt;
from skimage import transform
import numpy as np

import random

def load_data(data_directory) :
    directories = [d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory,d))]

    labels =[]
    images =[]

    for d in directories:
        label_directory = os.path.join(data_directory,d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.load(f))
            labels.append(int(d))
    return images, labels

ROOT_PATH = "C:/Users/Administrator/Downloads/"

train_data_directory= os.path.join(ROOT_PATH, "BelgiumTSC_Training/Training")
test_data_directory = os.path.join(ROOT_PATH, "BelgiumTSC_Testing/Testing")

images, labels = load_data(train_data_directory)

print(len(images))

images28 = [transform.resize(image, (28, 28)) for image in images]

images28 = np.array(images28)

gray = rgb2gray(images28)

#initializing the neural newtork

# Initialize placeholders
x = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28])
y = tf.placeholder(dtype = tf.int32, shape = [None])

# Flatten the input data
images_flat = tf.contrib.layers.flatten(x)

# Fully connected layer
logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

# Define a loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y,
                                                                     logits = logits))
# Define an optimizer
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# Convert logits to label indexes
correct_pred = tf.argmax(logits, 1)

# Define an accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


tf.set_random_seed(1234)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(201):
        _, loss_value = sess.run([train_op, loss], feed_dict={x: gray, y: labels})
        if i%10 ==0:
            print("Loss:" , loss)


# Pick 10 random images
sample_indexes = random.sample(range(len(gray)), 10)
sample_images = [gray[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]

sess = tf.Session()

sess.run(tf.global_variables_initializer())
predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]

# Print the real and predicted labels
print(sample_labels)
print(predicted)


# Display the predictions and the ground truth visually.
fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5, 2,1+i)
    plt.axis('off')
    color='green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction),
             fontsize=12, color=color)
    plt.imshow(sample_images[i],  cmap="gray")

plt.show()
# traffic_signs = [300, 2250, 3650, 4000]
#
# for i in range(len(traffic_signs)):
#     plt.subplot(1, 4, i+1)
#     plt.axis('off')
#     plt.imshow(gray[traffic_signs[i]], cmap = plt.get_cmap('gray'))
#     plt.subplots_adjust(wspace=0.5)
#
# plt.show()



# plt.figure(figsize=(15, 15))
#
# i =1
#
# unique_labels = set(labels)
# for label in unique_labels:
#     # You pick the first image for each label
#     image = images[labels.index(label)]
#     # Define 64 subplots
#     plt.subplot(8, 8, i)
#     # Don't include axes
#     plt.axis('off')
#     # Add a title to each subplot
#     plt.title("Label {0} ({1})".format(label, labels.count(label)))
#     # Add 1 to the counter
#     i += 1
#     # And you plot this first image
#     plt.imshow(image)
#
# plt.show()


#print(len(set(labels)))
#plt.hist(labels,62)
