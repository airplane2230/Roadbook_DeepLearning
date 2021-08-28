import tensorflow as tf

class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, init_lr, warmup_epoch,
                 decay_fn
                 ):
        self.init_lr = init_lr
        self.decay_fn = decay_fn
        self.warmup_epoch = warmup_epoch
        self.lr = 1e-4
        
    def get_config(self):
        # not working
        config = {
            'learning_rate': self.lr
        }
        
        return config
    
    # No Override
    def on_epoch_begin(self, epoch, logs = None):
        global_epoch = tf.cast(epoch + 1, tf.float64)
        warmup_epoch_float = tf.cast(self.warmup_epoch, tf.float64)
        
        lr = tf.cond(
            global_epoch < warmup_epoch_float,
            lambda: tf.cast(self.init_lr * (global_epoch / warmup_epoch_float), tf.float64),
            lambda: tf.cast(self.decay_fn(epoch - warmup_epoch_float), tf.float64)
        )
        
        self.lr = lr
        
    def __call__(self, step):
        self.on_epoch_begin(step, logs = None)
        
        return self.lr