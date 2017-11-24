import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
from matplotlib import pyplot as plt

class NN_approx:
    def __init__(self, inD, outD, layers):
        lr = 1e-4

        self.weights = []
        self.biases = []

        layers = [inD] + layers + [outD]

        for i in range(len(layers)-1):
            self.weights.append(tf.Variable(tf.truncated_normal(shape=(layers[i], layers[i+1]), stddev=0.1), name='w%d'%i))
            self.biases.append(tf.Variable(tf.constant(0.1, shape=(1, layers[i+1])), name='b%d'%i))

        self.X = tf.placeholder(tf.float32, shape=(None, inD), name='X')
        self.Y = tf.placeholder(tf.float32, shape=(None, outD), name='Y')

        # compute prediction and cost function
        Y_hat = self.X

        for i in range(len(layers)-2):
            Y_hat = tf.matmul(Y_hat, self.weights[i])
            Y_hat += self.biases[i]
            Y_hat = tf.nn.relu(Y_hat)

        Y_hat = tf.matmul(Y_hat, self.weights[-1]) + self.biases[-1]
        Y_hat_softmaxed = tf.nn.softmax(Y_hat)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=Y_hat))

        # optimisation
        self.train_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)
        self.predict_op = Y_hat_softmaxed

        init = tf.global_variables_initializer()
        self.session = tf.InteractiveSession()
        self.session.run(init)

    def partial_fit(self, X, Y):
        self.session.run(self.train_op, feed_dict={self.X: X, self.Y: Y})

    def predict(self, X):
        return self.session.run(self.predict_op, feed_dict={self.X: X})


if __name__ == '__main__':

    labeled_images = pd.read_csv('train.csv')
    images = labeled_images.iloc[0:10000, 1:]
    labels = labeled_images.iloc[0:10000, :1]
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.9,
                                                                            random_state=0)

    clf = NN_approx(28*28, 10, [500])
    print(images.shape)
    batch_size = 100
    n_epochs = 5
    for j in range(n_epochs):
        for i in range(int(train_images.shape[0]/batch_size)-1):
            img = train_images.iloc[i*batch_size:(i+1)*batch_size,:]._values
            outpt = tf.one_hot(np.array(train_labels[i*batch_size:(i+1)*batch_size]).reshape(batch_size), 10)
            outpt = clf.session.run(outpt)
            clf.partial_fit(img, outpt)

        print(j)

    correct =0

    for i in range(test_labels.shape[0]):
        img = train_images.iloc[i]._values
        y_hat = clf.predict([img])
        print('prediction:', np.argmax(y_hat[0]), 'real', train_labels.values[i])
        if np.argmax(y_hat[0]) == train_labels.values[i][0]:
            correct += 1
        # img = np.reshape(img, [28, 28])
        # imgplot = plt.imshow(img)
        # plt.show()

    print(correct/test_labels.shape[0])




