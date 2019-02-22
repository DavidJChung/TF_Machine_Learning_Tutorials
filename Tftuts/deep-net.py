import tensorflow as tf

'''
input gets weighted into hidden layer (avtivation function) gets weightd into hidden layer 2 
(activation function) gets weights into output layer            Basic feed forward

compare output to intended output and calculate cost funct (cross entropy - how close we are to our desired cost funct)
pass into optimization funct(optimizer) which will minize cost using AdamOptimizer

goes backwards and manipulates weight (back proppagation)

feed forward + back prop = 1 epoch or iteration will be done around 10-20 times
'''

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)       #usefull for multiclass

# 10 classes, 0-9
'''
one_hot does

0 = [1,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0,0]
3 = [0,0,0,1,0,0,0,0,0]

one bit is hot at a time to denote information

'''

#hidden layer node assignment

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size =100

x = tf.placeholder('float',[None,784])
y = tf.placeholder('float')

def neural_network_model(data):



    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}     #baised prevents 0 valued input data to fire neural network

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}

    l1=tf.add(tf.matmul(data,hidden_1_layer['weights']) ,hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']) , hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']) ,hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels=y))

    optimizer =tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epochs in range(hm_epochs):
            epoch_loss = 1
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epochs_x,epochs_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer,cost], feed_dict = {x: epochs_x, y: epochs_y})
                epoch_loss += c
            print('Epoch', epochs, 'completed out of', hm_epochs, 'loss: ', epoch_loss )

        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: ', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)