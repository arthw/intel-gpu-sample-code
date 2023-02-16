import numpy as np
import sys
import tensorflow as tf

from packaging import version
if version.parse(tf.keras.__version__.replace("-tf", "+tf")) < version.parse("2.11"):
    from tensorflow.keras import optimizers
else:
    from tensorflow.keras.optimizers import legacy as optimizers

def main():
    gpus = tf.config.experimental.list_physical_devices('XPU')
    print("### gpus ", gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[0], 'XPU')
    (mnist_images, mnist_labels), _ = \
        tf.keras.datasets.mnist.load_data(path='mnist-%d.npz' % 0)
    dataset = tf.data.Dataset.from_tensor_slices(
        ((mnist_images[..., tf.newaxis] / 255.0).astype(np.float32),
        tf.cast(mnist_labels, tf.int32)))

    dataset = dataset.repeat().shuffle(10000).batch(128)
    mnist_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
        tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Horovod: adjust learning rate based on number of GPUs.
    scaled_lr = 0.001 
    opt = optimizers.Adam(scaled_lr)

    mnist_model.compile(loss=tf.losses.SparseCategoricalCrossentropy(),
                        optimizer=opt,
                        metrics=['accuracy'],
                        experimental_run_tf_function=False)

    callbacks = [
    ]

    callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

    verbose = 1 

    # Train the model.
    mnist_model.fit(dataset, steps_per_epoch=500, callbacks=callbacks, epochs=24, verbose=verbose)

if __name__ == '__main__':
    main()
