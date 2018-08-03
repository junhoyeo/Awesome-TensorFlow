# -*- coding: utf-8 -*-
# Copyright 2018 The TensorFlow Authors
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# ===== Import the Fashion MNIST dataset =====
fashion_mnist = keras.datasets.fashion_mnist
# Fashion MNIST dataset을 사용 (10개 카테고리의 28*28 사이즈의 그레이스케일 이미지 70,000개)
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# 트레이닝 세트 : (train_images, train_labels)
# 테스트 세트 : (test_images, test_labels)

class_names = ['T-shirt/top', 'Trouser', 'Pull`over', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# 각각의 이미지가 매핑되는 라벨의 클래스 이름

# ===== Preprocess the data =====
train_images = train_images / 255.0
test_images = test_images / 255.0
# 데이터 전처리 (각 픽셀의 값 0~250을 0~1 사이의 float로 스케일)

# ===== Build the model =====
# 신경망 구축 -> 모델의 각 레이어를 설정 + 모델을 컴파일

## === Setup the layers ===
# 신경망의 기본 구성 요소는 각 레이어 -> 입력된 데이터에서 표현을 추출
# 딥러닝 -> 간단한 레이어 여러 개를 연결
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    # 이미지의 포맷을 2차원 배열(28*28)로 변환하는 레이어
    keras.layers.Dense(128, activation=tf.nn.relu),
    # 128개 노드로 이루어진 레이어
    keras.layers.Dense(10, activation=tf.nn.softmax)
    # 10개 노드로 이루어진 softmax 출력층 
])

## === Compile the model ===
# 최적화 함수(optimizer) : 모델이 업데이트되는 방식 -> 갱신 방법
# 손실 함수(loss) : 학습 도중 모델의 정확도를 측정 -> 모델을 올바른 방향으로 조종
# 측정 항목(metrics) : 학습 및 테스트 단계 모니터링에 사용
model.compile(
    optimizer=tf.train.AdamOptimizer(), # Adam 갱신 기법
    loss='sparse_categorical_crossentropy', # 교차 엔트로피 오차
    metrics=['accuracy'] # 정확도
)

# ===== Train the model =====
# 1. 학습 데이터를 모델에 제공 -> (train_images, train_labels)
# 2. 모델이 각각의 이미지와 라벨을 연결하는 방법을 학습
# 3. 테스트 세트에 대해서 예측 -> 테스트 세트의 라벨과 비교
model.fit(train_images, train_labels, epochs=5)

# ===== Evaluate accuracy =====
# 테스트 세트에 대한 예측의 정확도를 출력
# 테스트 데이타에 대한 정확도는 overfitting 때문에 트레이닝 데이터에 대한 정확도보다 약간 작음
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
# ('Test accuracy:', 0.8714)

# ===== Make predictions =====
# 이미지에 대해서 예측
predictions = model.predict(test_images)

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions[i])
    true_label = test_labels[i]
    if predicted_label == true_label:
      color = 'green' # 예측 결과가 정확하면 초록색 텍스트 라벨
    else:
      color = 'red' # 예측 결과가 잘못되었으면 빨간색 텍스트 라벨
    plt.xlabel("{} ({})".format(class_names[predicted_label], 
                                class_names[true_label]),
                                color=color)
plt.show() # 예측 결과 중 앞의 25개 정보를 출력
