import tensorflow.compat.v1 as tf
tf.disable_eager_execution()  # 필수! TF 1.x 방식 사용 가능하게 만듦
import keras
import pandas as pd

# Model Parameters
learning_rate = 0.001
num_epochs = 10
batch_size = 100

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize and preprocess data
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# One-hot encode labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Network Parameters
n_input = 784
n_hidden1 = 512
n_hidden2 = 256
n_classes = 10

# Placeholders
X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_classes])

# Weights & Biases
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden1])),
    'h2': tf.Variable(tf.random_normal([n_hidden1, n_hidden2])),
    'out': tf.Variable(tf.random_normal([n_hidden2, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden1])),
    'b2': tf.Variable(tf.random_normal([n_hidden2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Model
def multilayer_perceptron(x):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    return tf.matmul(layer_2, weights['out']) + biases['out']

# Construct model
logits = multilayer_perceptron(X)

# Loss & Optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Init
init = tf.global_variables_initializer()
total_batch = x_train.shape[0] // batch_size

# Dataset pipeline
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
iterator = tf.data.make_initializable_iterator(dataset)
next_batch = iterator.get_next()

# Session
with tf.Session() as sess:
    sess.run(init)
    sess.run(iterator.initializer)

    for epoch in range(num_epochs):
        avg_loss = 0.
        for i in range(total_batch):
            batch_x, batch_y = sess.run(next_batch)
            _, l = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
            avg_loss += l / total_batch

        acc = sess.run(accuracy, feed_dict={X: x_test, Y: y_test})
        print("Epoch: %02d Loss: %.4f Accuracy: %.4f%%" % (epoch+1, avg_loss, acc * 100))
        sess.run(iterator.initializer)

    print("Final Test Accuracy: %.4f%%" % (sess.run(accuracy, feed_dict={X: x_test, Y: y_test}) * 100))
    print("\nTensorflow:", tf.__version__)

    # 학생 정보 출력
    data = {
        '이름': ['최서영'],
        '학번': [2315737],
        '학과': ['인공지능공학부']
    }
    df = pd.DataFrame(data)
    print("\n", df, "\n")

    