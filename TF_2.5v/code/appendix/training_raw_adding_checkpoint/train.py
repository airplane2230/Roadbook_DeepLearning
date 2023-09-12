import tensorflow as tf
import numpy as np

from tensorflow.keras.datasets import mnist

from tqdm import tqdm
from lr_schedule import LRSchedule
from basic_CNN import get_model

tf.keras.backend.set_floatx('float64')
print(tf.__version__)

def make_datasets(x, y):
    # (28, 28) -> (28, 28, 1)
    def _new_axis(x, y):
        y = tf.one_hot(y, depth = 10)
        
        return x[..., tf.newaxis], y
            
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(_new_axis, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    ds = ds.shuffle(BATCH_SIZE * 2).batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    
    return ds

@tf.function
def train_step(inp, tar, training = True):
    with tf.GradientTape() as tape:
        predictions = model(inp)
        loss = loss_function(tar, predictions)
        
    if training:
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_accuracy.update_state(tar, predictions)
    
    return loss, predictions

def get_steps(x_size, batch_size):
    if x_size / batch_size == 0:
        return x_size // batch_size
    else:
        return x_size // batch_size + 1


if __name__ == "__main__":
    # hyperparameter
    EPOCHS = 1000
    BATCH_SIZE = 512

    init_epoch = 1
    warmup_epoch = 5
    
    init_lr = 1e-3
    min_lr = 1e-6
    power = 1.
    
    isSchedule = True
    
    init_path = './model/init/'
    ckpt_path = './model/ckpt/'


    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    train_ds = make_datasets(x_train, y_train)

    model = get_model() # model.summary()

    lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate = init_lr,
        decay_steps = EPOCHS - warmup_epoch,
        end_learning_rate = min_lr,
        power = power
    )
    
    # get init epoch for training continued
    init_epoch_ckpt = tf.train.Checkpoint(epoch = tf.Variable(1))
    init_epoch_ckpt_manager = tf.train.CheckpointManager(init_epoch_ckpt, ckpt_path, max_to_keep = 5)
    init_epoch_ckpt.restore(init_epoch_ckpt_manager.latest_checkpoint).expect_partial()
    
    if init_epoch_ckpt_manager.latest_checkpoint:
        print(f'######### Continue Training!, {init_epoch_ckpt.epoch.numpy()}')
        init_epoch = init_epoch_ckpt.epoch.numpy()

    if isSchedule:
        optimizer = tf.keras.optimizers.Adam(learning_rate = LRSchedule(init_lr,
                                                                        warmup_epoch = warmup_epoch,
                                                                        steps_per_epoch = get_steps(x_train.shape[0], BATCH_SIZE),
                                                                        decay_fn = lr_scheduler,
                                                                        continue_epoch = init_epoch)
                                            )
    else:
        optimizer = tf.keras.optimizers.Adam()

#     init_ckpt = tf.train.Checkpoint(model = model, optimizer = optimizer)

#     init_ckpt_manager = tf.train.CheckpointManager(init_ckpt, init_path, max_to_keep = 10)
#     init_ckpt_manager.save()
#     print('save init!')
    
    loss_plot = []
    loss_function = tf.keras.losses.CategoricalCrossentropy()

    train_accuracy = tf.keras.metrics.CategoricalAccuracy()

    ckpt = tf.train.Checkpoint(epoch = tf.Variable(1), loss = tf.Variable(1., dtype = tf.float64),
                               accuracy = tf.Variable(1., dtype = tf.float64),
                               model = model,
                               optimizer = optimizer,
                               )
    ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep = 10)

    status = ckpt.restore(ckpt_manager.latest_checkpoint)
    
    if ckpt_manager.latest_checkpoint:
        print(f'Restored from {ckpt_manager.latest_checkpoint}')
        print(f'epoch: {ckpt.epoch.numpy()}, accuracy: {ckpt.accuracy.numpy()}, loss: {ckpt.loss.numpy()}')
    else:
        print('Initializing from Scratch')
    
    for epoch in range(init_epoch, EPOCHS):
        total_loss = 0.
        train_accuracy.reset_states()

        tqdm_dataset = tqdm(enumerate(train_ds))
        
        if isSchedule:
            print('current learning_rate: ', optimizer.lr(optimizer.iterations))
#             print(optimizer.get_config())

        for (batch, (tensor, target)) in tqdm_dataset:
            batch_loss, predictions = train_step(tensor, target, training = True)
            total_loss += batch_loss

            total_loss_format = total_loss / (batch + 1)

            tqdm_dataset.set_postfix({
                'Epoch': epoch,
                'Loss': '{:.4f}'.format(batch_loss.numpy()),
                'Total Loss': '{:.4f}'.format(total_loss_format),
                'Accuracy': '{:.4f}'.format(train_accuracy.result().numpy())
            })

        loss_plot.append(total_loss_format)

        if np.min(loss_plot) == loss_plot[-1]:
            print(f'{epoch} - loss: {loss_plot[-1]:.4f} - accuracy: {train_accuracy.result().numpy():.4f}')
            ckpt.epoch.assign(epoch)
            ckpt.accuracy.assign(train_accuracy.result().numpy())
            ckpt.loss.assign(total_loss_format)
            ckpt_manager.save()
