import tensorflow as tf
import pandas as pd

from model.modeling import get_model

# 데이터를 불러오고, Dataset 객체를 만듭니다.
train_df = pd.read_csv("./csv_data/nocolorinfo/train.csv")
train_slices = tf.data.Dataset.from_tensor_slices(dict(train_df))

# {image, black, blue, ...}
# 주석을 풀고, Dataset 객체가 출력하는 데이터를 확인하세요.
# for feature_batch in train_slices.take(1):
#     print(feature_batch)

def precessing(data_dict):
    img = tf.io.read_file(data_dict["image"])
    img = tf.image.decode_jpeg(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [112, 112])

    label = list(data_dict.values())[1:]

    return img, label

# map() 함수를 통해 precessing 함수를 적용합니다.
clothes_ds = train_slices.map(precessing)

# 버퍼에 100씩 저장하고, 임의로 32개의 요소를 꺼내서 사용합니다.
clothes_ds = clothes_ds.shuffle(100).batch(32)

# 모델을 불러오고, 학습합니다.
model = get_model()
model.fit(clothes_ds, epochs=2)