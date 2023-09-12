import tensorflow as tf

class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, init_lr, warmup_epoch, steps_per_epoch,
                 decay_fn, *,
                 continue_epoch = 1
                 ):
        self.init_lr = init_lr
        self.lr = init_lr
        
        self.decay_fn = decay_fn
        
        self.warmup_epoch = warmup_epoch
        self.steps_per_epoch = steps_per_epoch
        self.continue_epoch = continue_epoch
    
    # No Override
    def on_epoch_begin(self, epoch, logs = None):
        epoch = tf.cast(epoch, tf.float64)
        
        global_epoch = tf.cast(epoch, tf.float64)
        warmup_epoch_float = tf.cast(self.warmup_epoch, tf.float64)
        
        lr = tf.cond(
            global_epoch < warmup_epoch_float,
            lambda: tf.cast(self.init_lr * (global_epoch / warmup_epoch_float), tf.float64),
            lambda: tf.cast(self.decay_fn(epoch - warmup_epoch_float), tf.float64)
        )
        
        self.lr = lr
        
    def __call__(self, step):
        def compute_epoch(step):
            return step // self.steps_per_epoch

        epoch = compute_epoch(step)
        epoch = epoch + self.continue_epoch

        self.on_epoch_begin(epoch, logs = None)

        return self.lr