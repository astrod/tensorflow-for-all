# -*- coding: utf-8 -*-

# 절대 임포트 설정
from __future__ import absolute_import
from __future__ import print_function

# 필요한 라이브러리들을 임포트
import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# 필요한 데이터를 다운로드한다.
url = 'http://mattmahoney.net/dc/'

# 파일이 존재하지 않으면 다운로드한다.
# 파일의 사이즈가 적절한지 체크한다.
def download(filename, expected_bytes):
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('파일이 존재하고, 파일의 크기가 적확하다', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            '파일을 검사하는데 실패했다. ' + filename
        )
    return filename

# 위키피디아의 테스트 데이터를 다운로드한다.
# http://mattmahoney.net/dc/textdata.html 의 Relationship of Wikipedia Text to Clean Text 참조
filename = download('text8.zip', 31344016)

# zip 파일 압축을 해제하고, 단어들의 리스트를 읽는다.
def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data =f.read(f.namelist()[0]).split()
    return data

# words 에는 모든 단어들의 리스트가 들어있다. 중복은 제거되지 않음
words = read_data(filename)
print('Data size', len(words))

# dictionaray를 만든다. UNK 토큰을 가지고 노출빈도가 잦지 않은 단어들을 교체한다.
vocabulary_size = 50000

def build_dataset(words):
    count = [['UNK', -1]]
    # 가장 노출 빈도수가 높은 단어를 50000개까지 추린다.
    # count 배열은 다음과 같은 형태로 나타난다.
    # [('desylva', 9), ('uplink', 9), ('tendon', 12) .....]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)

    # dictionary : 'infighting': 23487, 'buena': 26390, 'agra': 25865 이런 형태의 데이터가 됨. 뒤의 숫자는 인덱스(노출빈도수가 아님)

    data = list()
    unk_count = 0
    for word in words:
        # 위에서 세팅한 값으로 인덱스를 잡아준다.
        if word in dictionary:
            index = dictionary[word]
        # 사전에 값이 없는 경우는 UNK 토큰으로 값을 입력한다.
        else:
            index = 0 # dictionary['UNK']
            unk_count += 1
        # data 리스트는 전체 word 인덱스의 집합이 된다. UNK는 0으로 입력된다.
        data.append(index)
    count[0][1] = unk_count # unk가 몇개 있는지 계산하여 값을 입력한다.
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
del words # 이제 워드 사전은 필요없으므로, 메모리에서 지운다.

# Most common words (+UNK) [['UNK', 418391], (b'the', 1061396), (b'of', 593677), (b'and', 416629), (b'one', 411764)]
print('Most common words (+UNK)', count[:5])

# Sample data [5234, 3082, 12, 6, 195, 2, 3136, 46, 59, 156] [b'anarchism', b'originated', b'as', b'a', b'term', b'of', b'abuse', b'first', b'used', b'against']
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0
skip_window = 1       # 윈도우 크기 : 왼쪽과 오른쪽으로 얼마나 많은 단어를 고려할지를 결정.
num_skips = 2         # 레이블(label)을 생성하기 위해 인풋을 얼마나 많이 재사용 할 것인지를 결정.

# Step 3: skip-gram model을 위한 트레이닝 데이터(batch)를 생성하기 위한 함수.
# batch_size : 배치 데이터의 크기
# num_skips : 몇 개로 배치 묶음을 할지 결정하는 값. 예를 들면
# [1, 2, 3, 4, 5] 가 있을때, skip_window가 1 이면 앞뒤로 한개씩 윈도우가 움직인다. [1,2,3], [2,3,4] 이렇게
# 하지만, 실제로 코드에서는 [1, 2], [2, 3], [3, 4] 와 같이 데이터를 묶는다. 이 값을 몇개 단위로 묶을지 결정하는 파라미터
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1 # [ skip_window target skip_window ]
    # 데크 : 큐인데 양방향에서 데이터를 삭제할 수 있는 자료구조. append 하면 오른쪽에 데이터를 추가할 수 있다.
    # mexlen을 넘겨서 append 하면, 선입된 데이터가 자연스럽게 사라진다(큐와 같음)
    buffer = collections.deque(maxlen=span)

    # data 전체를 돌면서 해당 데이터를 buffer에 쌓는다. span의 크기만큼 쌓음. buffer init
    # deque([5234, 3081, 12], maxlen=3)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data) # data_index + 1 이 리턴됨(오버플로우 되는 경우 막기 위해)
        
    for i in range(batch_size // num_skips): # // : 나누기 연산 후 소수점 이하의 수를 버리고, 정수 부분의 수만 구함
        # print("start buffer : ", buffer)
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [ skip_window ]
        for j in range(num_skips):
            # [0, 1, 2] 중에, target이 아닌걸 뽑음(span 이 2 이므로, 0,1,2 중에 하나가 리턴되는데 리턴값이 target과 다른게 나올때까지 돌린다)
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            # targets_to_avoid :  [1, 2]
            # targets_to_avoid :  [1, 2, 0]
            # append 하면 targets_to_avoid 에 skip_window, skip_window +- 1 인 것 두개가 들어가게 된다.
            targets_to_avoid.append(target) 
            # print("targets_to_avoid : ", targets_to_avoid)
            batch[i * num_skips + j] = buffer[skip_window] # 해당 데이터 insert. skip-gram 방식이므로 타겟이 배치 입력값이고 컨텍스트가 출력값이다.
            labels[i * num_skips + j, 0] = buffer[target] # 컨텍스트 값이 들어감
        # 위에서 num_skips 로 두번 돌면, 버퍼에 대해 두번 작업이 완료된 것임
        # print("before append buffer : ", buffer)
        # print("append data : ", data[data_index])
        # data 에서 새로운 친구를 데큐에 넣는다. 맨 앞에 것이 자연스럽게 밀려남
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
        # print("end buffer : ", buffer)
    return batch, labels  

batch, labels = generate_batch(batch_size=8, num_skips=num_skips, skip_window=skip_window)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]],
      '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# Step 4: skip-gram model 만들고 학습시킨다.

batch_size = 128
embedding_size = 128  # embedding vector의 크기. 하나의 데이터를 어느정도 단위로 시각화할지 결정

# sample에 대한 validation set은 원래 랜덤하게 선택해야한다. 하지만 여기서는 validation samples을 
# 가장 자주 생성되고 낮은 숫자의 ID를 가진 단어로 제한한다.
valid_size = 16     # validation 사이즈.
valid_window = 100  # 분포의 앞부분(head of the distribution)에서만 validation sample을 선택한다.
valid_examples = np.random.choice(valid_window, valid_size, replace=False) # valid_window 만큼의 숫자 내에서, valid_size 만큼의 랜덤 함수를 만든다.
num_sampled = 64    # sample에 대한 negative examples의 개수.

graph = tf.Graph()

with graph.as_default():
  # 트레이닝을 위한 인풋 데이터들
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # embedding vectors 행렬을 랜덤값으로 초기화
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        # 행렬에 트레이닝 데이터를 지정
        # skip_gram 프로세스에서, 입력 벡터를 만드는 과정이다. 입력 벡터를 만드는 과정이기 때문에, 문자가 128 차원의 백터로 전환된 임베딩 데이터가 이 곳에 저장되게 된다.
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # NCE loss를 위한 변수들을 선언
        # full connected 이기 때문에, 단어장 사이즈 * 임베딩 벡터 사이즈 만큼의 weight가 필요하다.
        # truncated_normal : 절단정규분포로부터의 난수값을 반환합니다. 생성된 값들은 평균으로부터 떨어진 버려지고 재선택된 두 개의 표준편차보다 큰 값을 제외한 지정된 평균과 표준 편차를 가진 정규 분포를 따릅니다.
        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # batch의 average NCE loss를 계산한다.
    # tf.nce_loss 함수는 loss를 평가(evaluate)할 때마다 negative labels을 가진 새로운 샘플을 자동적으로 생성한다.
    # 내부에서 one-hot 인코딩 방식으로 입력값을 변환하지 않을까 추측됨
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights, # 웨이트
                     biases=nce_biases, # 편향
                     labels=train_labels, # 트레이닝 라벨
                     inputs=embed, # 임베딩 벡터
                     num_sampled=num_sampled, # 네거티브 샘플의 개수
                     num_classes=vocabulary_size)) # 전체 단어장 크기

    # SGD optimizer를 생성한다.
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # minibatch examples과 모든 embeddings에 대해 cosine similarity를 계산한다.
    # 실제로 두 값이 얼마나 가까이 있는지는 벡터의 거리보다는, 코사인 유사도를 가지고 값을 구한다.
    # 벨리드 데이터셋으로 추출된 16개 단어가 전체 단어중(normalized_embeddings) 어떤 단어와 유사도가 있는지 구하는 코드
    # https://ko.wikipedia.org/wiki/%EC%BD%94%EC%82%AC%EC%9D%B8_%EC%9C%A0%EC%82%AC%EB%8F%84
    # http://euriion.com/?p=548
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    # normalized_embeddings :  Tensor("truediv:0", shape=(50000, 128), dtype=float32)
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset) # 16개를 뽑기 때문에, 결과값이 50000 * 16
    # similarity :  Tensor("MatMul_1:0", shape=(16, 50000), dtype=float32)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True) # 임베딩 백터와 valid 데이터셋을 추출하여, 두 임베딩 백터의 코사인 유사성을 계산한다.

    # Step 5: 트레이닝을 시작한다.
    num_steps = 100001

    with tf.Session(graph=graph) as session:
        # 트레이닝을 시작하기 전에 모든 변수들을 초기화한다.
        tf.initialize_all_variables().run()
        print("Initialized")

        average_loss = 0
        for step in xrange(num_steps):
            batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
            feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

            # optimizer op을 평가(evaluating)하면서 한 스텝 업데이트를 진행한다.
            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                # 평균 손실(average loss)은 지난 2000 배치의 손실(loss)로부터 측정된다.
                print("Average loss at step ", step, ": ", average_loss)
                average_loss = 0

            # Note that this is expensive (~20% slowdown if computed every 500 steps)
            # valid 데이터셋으로 추출된 단어들이 어떤 단어와 유사한지 보여주는 코드
            if step % 10000 == 0:
                sim = similarity.eval()
                for i in xrange(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8 # nearest neighbors의 개수
                    # 코사인 유사도는 -1 이면 반대, 0이면 독립, 1이면 같은거라서 작은 수부터 추리기 위해 -를 붙여준다.
                    # 정렬한 값의 인덱스를 리턴함. 가장 가까운 값들의 인덱스가 됨. 가장 값이 적은 애를 8개 추려서 리턴한다.
                    nearest = (-sim[i, :]).argsort()[1:top_k+1] 
                    log_str = "Nearest to %s:" % valid_word
                    for k in xrange(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = "%s %s," % (log_str, close_word)
                    print(log_str)
        final_embeddings = normalized_embeddings.eval()

# Step 6: embeddings을 시각화한다.

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  #in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]
        plt.scatter(x, y)
        plt.annotate(label,
                    xy=(x, y),
                    xytext=(5, 2),
                    textcoords='offset points',
                    ha='right',
                    va='bottom')

    plt.savefig(filename)

try:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 500
    # T-SNE 를 사용하여 128 차원을 2차원으로 축소하여 표현한다.
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
    labels = [reverse_dictionary[i] for i in xrange(plot_only)]
    plot_with_labels(low_dim_embs, labels)

except ImportError:
    print("Please install sklearn and matplotlib to visualize embeddings.")
