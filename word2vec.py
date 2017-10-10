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
vocabularay_size = 50000

def build_dataset(words):
    count = [['UNK', -1]]
    # 가장 노출 빈도수가 높은 단어를 50000개까지 추린다.
    # count 배열은 다음과 같은 형태로 나타난다.
    # [('desylva', 9), ('uplink', 9), ('tendon', 12) .....]
    count.extend(collections.Counter(words).most_common(vocabularay_size - 1))
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
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1 # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data) # data_index + 1 이 리턴됨(오버플로우 되는 경우 막기 위해)
    for i in range(batch_size // num_skips): # // : 나누기 연산 후 소수점 이하의 수를 버리고, 정수 부분의 수만 구함
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [ skip_window ]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels  

batch, labels = generate_batch(batch_size=8, num_skips=num_skips, skip_window=skip_window)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]],
      '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

