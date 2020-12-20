import wandb
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def train(data,epochs):
    data = tf.data.Dataset.from_tensor_slices(data)
    data = data.shuffle(1024,reshuffle_each_iteration=True).batch(32,drop_remainder=True).repeat()
    model = GAN()
    wandb.init()
    loss = {
        "g_loss":0,
        "d_loss":0,
    }

    for i, images in enumerate(tqdm(data)):
        logs = train_step(model, images)
        loss["g_loss"]+= logs["g_loss"]
        loss["d_loss"]+= logs["d_loss"]
        if i % 1000 == 0:
            loss["g_loss"]/= 1000.0
            loss["d_loss"]/= 1000.0
            wandb.log(loss) 
            loss["g_loss"] = 0
            loss["d_loss"] = 0
        if i % 10000 == 0:
            wandb.log({"generated":[wandb.Image(logs["images"][i]) for i in range(logs["images"].shape[0])]})

@tf.function
def train_step(model,images):
    images = tf.image.random_crop(images,[8,64,64,3])
    images = tf.image.random_flip_up_down(images)
    images = tf.image.random_flip_left_right(images)
    images = tf.image.per_image_standardization(tf.cast(images,tf.float32))

    inputs = tf.random.normal([images.shape[0],128])
    
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
    return {"d_loss": d_loss, "g_loss": g_loss,"images": generated}

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
        
def get_generator():
    generator = keras.Sequential(
        [
            layers.Input(shape = (128)),
            layers.Dense(8*8*256, use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Reshape((8,8,256)),
            layers.Conv2DTranspose(256, 4, strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(256, 4, strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(256, 4, strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(3, 3, padding="same"),
        ],name="generator")

    generator.summary()
    return generator

def get_discriminator():
    discriminator = keras.Sequential(
    [
        keras.Input(shape=(64, 64, 3)),
        layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.GlobalMaxPooling2D(),
        layers.Dense(1),
    ],name="discriminator")

    discriminator.summary()
    return discriminator