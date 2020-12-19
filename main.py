import torch
import tensorflow as tf
import numpy as np
from tf_gan import GAN,WandbLogs
import matplotlib.pyplot as plt
def tensorflow(data):
    data = tf.data.Dataset.from_tensor_slices(data)
    data = data.shuffle(1024,reshuffle_each_iteration=True).batch(32,drop_remainder=True)
    model = GAN()
    model.fit(data,epochs=100,callbacks=[WandbLogs()])  

if __name__ == "__main__":
    data = np.load("/kaggle/input/noel-dataset/noel.npy")
    tensorflow(data)