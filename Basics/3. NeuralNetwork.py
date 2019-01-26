import numpy as np
import tensorflow as tf

x_data = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)
y_label = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)
m = tf.Variable(0.44)
b = tf.Variable(0.87)

error = 0
for x,y in zip(x_data,y_label):
    y_hat = m*x + b  #Our predicted value
    error += (y-y_hat)**2 # The cost we want to minimize (we'll need to use an optimization function for the minimization!)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error) # changes the values of Variables

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    epochs = 100
    for i in range(epochs):
        sess.run(train)

    final_slope , final_intercept = sess.run([m,b])
    fw = tf.summary.FileWriter('tf_graph',sess.graph)