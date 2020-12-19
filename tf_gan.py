import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import wandb
class GAN(keras.Model):
    def __init__(self):
        super(GAN, self).__init__()
        self.discriminator = get_discriminator()
        self.generator = get_generator()
        self.compile()

    def compile(self):
        super(GAN, self).compile()
        self.d_optimizer = keras.optimizers.Adam(learning_rate=0.0003)
        self.g_optimizer = keras.optimizers.Adam(learning_rate=0.0003)
        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)

    @tf.function
    def train_step(model,images):
        images = tf.image.random_crop(images,[8,64,64,3])
        images = tf.image.per_image_standardization(tf.cast(images,tf.float32))
        inputs = tf.random.normal([images.shape[0],8,8,3])
        
        g_labels = tf.zeros((images.shape[0], 1))
        d_labels = tf.concat([tf.ones((images.shape[0], 1)), tf.zeros((images.shape[0], 1))], axis=0)

        with tf.GradientTape(persistent=True) as tape:
            generated = model.generator(inputs) 
            predictions = model.discriminator(generated)
            g_loss = model.loss_fn(g_labels, predictions)

            combined = tf.concat([generated,images],axis=0)
            predictions = model.discriminator(combined)
            d_loss = model.loss_fn(d_labels, predictions)

        grads = tape.gradient(g_loss, model.generator.trainable_weights)
        model.g_optimizer.apply_gradients(zip(grads, model.generator.trainable_weights))

        grads = tape.gradient(d_loss, model.discriminator.trainable_weights)
        model.d_optimizer.apply_gradients(zip(grads, model.discriminator.trainable_weights))
        return {"d_loss": d_loss, "g_loss": g_loss}

def get_generator():
    generator = keras.Sequential(
        [
            keras.Input(shape=(8,8,3)),
            # We want to generate 128 coefficients to reshape into a 7x7x128 map
            layers.Conv2DTranspose(128, 4, strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(128, 4, strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(128, 4, strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(3, 3, padding="same", activation="sigmoid"),
        ],name="generator")

    generator.summary()
    return generator

def get_discriminator():
    discriminator = keras.Sequential(
    [
        keras.Input(shape=(64, 64, 3)),
        layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.GlobalMaxPooling2D(),
        layers.Dense(1),
    ],name="discriminator")

    discriminator.summary()
    return discriminator

class WandbLogs(keras.callbacks.Callback):
    def __init__(self):
        wandb.init(entity="azeru",project="tensorflow")
    def on_epoch_end(self,epoch,logs=None):
        wandb.log(logs,step=epoch)