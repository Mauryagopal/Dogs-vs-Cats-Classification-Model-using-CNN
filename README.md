# 🧠 Image Classification using CNN (Accuracy: 90%)

This project implements a Convolutional Neural Network (CNN) model using TensorFlow and Keras to classify images into binary categories. The architecture is optimized for accuracy and achieves around **90% accuracy** on the dataset.

## 📝 Table of Contents

- [Model Overview](#model-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Training Configuration](#training-configuration)
- [Results](#results)
- [License](#license)

---

## 🧩 Model Overview

This CNN architecture uses multiple convolutional and pooling layers followed by dense layers to classify images. The final output layer uses a sigmoid activation, making it suitable for **binary classification tasks**.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout

model = Sequential()
model.add(Input(shape = (128,128,3)))

model.add(Conv2D(16, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=2))

model.add(Conv2D(32, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=2))

model.add(Conv2D(12, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(1, activation='sigmoid'))
