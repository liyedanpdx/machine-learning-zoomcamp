import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.applications.xception import decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("="*50)
print("GPU 信息:")
print(tf.config.list_physical_devices('GPU'))
print("="*50)

# 加载模型（第一次会下载权重文件）
print("正在加载 Xception 模型...")
model = Xception(weights='imagenet', input_shape=(299, 299, 3))
print("✅ 模型加载成功！")
print(f"模型参数量: {model.count_params():,}")