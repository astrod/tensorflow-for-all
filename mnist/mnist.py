# 신경망 구성을 손쉽게 해 주는 유틸리티 모음인 tensorflow.layers 를 사용해봅니다.
# 01 - CNN.py 를 재구성한 것이니, 소스를 한 번 비교해보세요.
# 이처럼 TensorFlow 에는 간단하게 사용할 수 있는 다양한 함수와 유틸리티들이 매우 많이 마련되어 있습니다.
# 다만, 처음에는 기본적인 개념에 익숙히지는 것이 좋으므로 이후에도 가급적 기본 함수들을 이용하도록 하겠습니다.
import tensorflow as tf

# kaggle 데이터 불러오기  
# filename_queue = tf.train.string_input_producer(
#     ['train.csv'], shuffle=False, name='filename_queue')

# reader = tf.TextLineReader()
# key, value = reader.read(filename_queue)

# # Default values, in case of empty columns. Also specifies the type of the
# # decoded result.
# #record_defaults = [[0.]] * 28
# xy = tf.decode_csv(value, record_defaults=record_defaults)

# train_x, train_y = tf.train.batch([xy[1:], xy[0]], batch_size = 1000)


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

#########
# 신경망 모델 구성
######
global_step = tf.Variable(0, trainable=False, name='global_step')

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])
is_training = tf.placeholder(tf.bool)

# 기본적으로 inputs, outputs size, kernel_size 만 넣어주면
# 활성화 함수 적용은 물론, 컨볼루션 신경망을 만들기 위한 나머지 수치들은 알아서 계산해줍니다.
# 특히 Weights 를 계산하는데 xavier_initializer 를 쓰고 있는 등,
# 크게 신경쓰지 않아도 일반적으로 효율적인 신경망을 만들어줍니다.
with tf.name_scope('conv1'):
    L1 = tf.layers.conv2d(X, 32, [3, 3])
    L1 = tf.layers.max_pooling2d(L1, [2, 2], [2, 2])
    L1 = tf.layers.dropout(L1, 0.7, is_training)

with tf.name_scope('conv2'):
    L2 = tf.layers.conv2d(L1, 64, [3, 3])
    L2 = tf.layers.max_pooling2d(L2, [2, 2], [2, 2])
    L2 = tf.layers.dropout(L2, 0.7, is_training)

with tf.name_scope('fullConnected1'):
    L3 = tf.contrib.layers.flatten(L2)
    L3 = tf.layers.dense(L3, 256, activation=tf.nn.relu)
    L3 = tf.layers.dropout(L3, 0.5, is_training)

with tf.name_scope('output'):
    model = tf.layers.dense(L3, 10, activation=None)

with tf.name_scope('optimizer'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost, global_step=global_step)
    tf.summary.scalar('cost', cost)

#########
# 신경망 모델 학습
######

sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())

ckpt = tf.train.get_checkpoint_state('./model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs', sess.graph)

batch_size = 100
total_batch = int(mnist.train.num_examples/batch_size)

for epoch in range(15):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape(-1, 28, 28, 1)
        _, cost_val = sess.run([optimizer, cost],
                               feed_dict={X: batch_xs,
                                          Y: batch_ys,
                                          is_training: True})

        summary = sess.run(merged, feed_dict={X: batch_xs, Y: batch_ys, is_training: True})
        writer.add_summary(summary, global_step=sess.run(global_step))

        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1),
          'Avg. cost =', '{:.4f}'.format(total_cost / total_batch))

saver.save(sess, './model/dnn.ckpt', global_step=global_step)
print('최적화 완료!')

#########
# 결과 확인
######
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도:', sess.run(accuracy,
                        feed_dict={X: mnist.test.images.reshape(-1, 28, 28, 1),
                                   Y: mnist.test.labels,
                                   is_training: False}))