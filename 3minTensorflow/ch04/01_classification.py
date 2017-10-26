import tensorflow as tf
import numpy as np

# [털, 날개]
x_data = np.array([[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]])

# 실측값. one-hot 인코딩 방식으로 표현됨
y_data = np.array([
    [1, 0, 0],  # 기타
    [0, 1, 0],  # 포유류
    [0, 0, 1],  # 조류
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1]
])

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([2, 3], -1., 1.))
b = tf.Variable(tf.zeros([3]))

L = tf.add(tf.matmul(X, W), b)
L = tf.nn.relu(L)

# 예측값
model = tf.nn.softmax(L)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis=1))

# 경사하강법 사용
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

# 세션 초기화
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 학습 진행
for step in range(100):
    sess.run(train_op, feed_dict={X : x_data, Y : y_data})

    if (step + 1) % 10 == 0:
        print(step + 1, sess.run(cost, feed_dict={X:x_data, Y:y_data}))

# 결과 예측
prediction = tf.argmax(model, axis=1)
target = tf.argmax(Y, axis=1)
print('예측값:', sess.run(prediction, feed_dict={X:x_data}))
print('실제값', sess.run(target, feed_dict={Y : y_data}))

# 정확도 출력
is_corrent = tf.equal(prediction, target) # 맞추면 true / 틀리면 false 값으로 배열을 변경 [true, false, true, false, false]
accuracy = tf.reduce_mean(tf.cast(is_corrent, tf.float32)) # false는 0, true는 1로 변경하여 1차원으로 합한 평균 => 2 /5 = 0.4
print('정확도 : %2f' % sess.run(accuracy * 100, feed_dict={X:x_data, Y:y_data})) # 100을 곱하면 확률이 됨