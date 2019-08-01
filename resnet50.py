import os
import errno
import argparse
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd

from tensorflow import keras

from read_tfrecord import tfrecord_train_input_fn

def main(_):
    # Horovod: initialize Horovod.
    hvd.init()

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    tf.compat.v1.enable_eager_execution(config=config)

    mnist_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, [3, 3], activation='relu'),
        tf.keras.layers.Conv2D(16, [3, 3], activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10)
    ])

    # Horovod: adjust learning rate based on number of GPUs.
    opt = tf.compat.v1.train.AdamOptimizer(0.001 * hvd.size())
    
    print("------------------HELLO-----------------")
    # Load the TFRecord dataset
    filename = "/mnt/bb/shutoaraki/train-00000-of-01024.tfrecords"
    dataset = tfrecord_train_input_fn(filename, batch_size=128)
    print("\n\n\nDATASET is right here!!!")
    print(dataset)
    print("\n\n\n")

    checkpoint_dir = './checkpoints'
    step_counter = tf.compat.v1.train.get_or_create_global_step() 
    checkpoint = tf.train.Checkpoint(model=mnist_model, optimizer=opt,
                                     step_counter=step_counter)

    # Horovod: adjust number of steps based on number of GPUs.
    for (batch, (images, labels)) in enumerate(
            dataset.take(20000 // hvd.size())):
        with tf.GradientTape() as tape:
            logits = mnist_model(images, training=True)
            loss_value = tf.losses.sparse_softmax_cross_entropy(labels, logits)

        # Horovod: add Horovod Distributed GradientTape.
        tape = hvd.DistributedGradientTape(tape)
        print("\n\n=====================================================")
        print(batch, (images, labels))
        print("=====================================================\n\n")
        grads = tape.gradient(loss_value, mnist_model.variables)
        opt.apply_gradients(zip(grads, mnist_model.variables),
                            global_step=tf.train.get_or_create_global_step())

        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        if batch == 0:
            hvd.broadcast_variables(mnist_model.variables, root_rank=0)
            hvd.broadcast_variables(opt.variables(), root_rank=0)

        if batch % 10 == 0 and hvd.local_rank() == 0:
            print('Step #%d\tLoss: %.6f' % (batch, loss_value))

    # Horovod: save checkpoints only on worker 0 to prevent other workers from
    # corrupting it.
    if hvd.rank() == 0:
        checkpoint.save(checkpoint_dir)



if __name__ == "__main__":
    tf.app.run()
