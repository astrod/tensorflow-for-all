import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

filename_queue = tf.train.string_input_producer(
    ['creditcard.csv'], shuffle=False, name='filename_queue')

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[0.]] * 31
xy = tf.decode_csv(value, record_defaults=record_defaults)

# collect batches of csv in
train_x_batch, train_y_batch = \
    tf.train.batch([xy[1:-1], xy[-1:]], batch_size=1000)

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 29])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([29, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.sigmoid(tf.matmul(X,W) + b)

# predicted
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# Start populating the filename queue.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(20):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_batch, Y: y_batch})
    # if step % 10 == 0:
    print(step, "Cost: ", cost_val)

coord.request_stop()
coord.join(threads)



# test
filename_queue = tf.train.string_input_producer(
    ['creditcard-test.csv'], shuffle=False, name='filename_queue')

test_reader = tf.TextLineReader()
test_key, test_value = test_reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
test_xy = tf.decode_csv(test_value, record_defaults=record_defaults)

# collect batches of csv in
test_x_batch, test_y_batch = \
    tf.train.batch([test_xy[1:-1], test_xy[-1:]], batch_size=100)

# Launch the graph in a session.
batch_sess = tf.Session()
# Initializes global variables in the graph.
batch_sess.run(tf.global_variables_initializer())

# Start populating the filename queue.
test_coord = tf.train.Coordinator()
test_threads = tf.train.start_queue_runners(sess=batch_sess, coord=test_coord)

for step in range(30):
    result_x_batch, result_y_batch = batch_sess.run([test_x_batch, test_y_batch])
    print("5")
    predicted = sess.run(
        [predicted], feed_dict={X: result_x_batch, Y: result_y_batch})

    print("predicted = ", predicted)
    

test_coord.request_stop()
test_coord.join(threads)