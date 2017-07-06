import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

## The following flags are related to save paths, tensorboard out6puts and screen outputs

tf.app.flags.DEFINE_string('version', 'cremi_adam_exp_decay_valA', '''A version number defining the directory to save
logs and checkpoints''')
tf.app.flags.DEFINE_integer('report_freq', 391, '''Steps takes to output errors on the screen
and write summaries''')
tf.app.flags.DEFINE_float('train_ema_decay', 0.95, '''The decay factor of the train error's
moving average shown on tensorboard''')


## The following flags define hyper-parameters regards training

tf.app.flags.DEFINE_integer('train_steps', 25000, '''Total steps that you want to train''')
tf.app.flags.DEFINE_boolean('is_full_validation', True, '''Validation w/ full validation set or
a random batch''')
tf.app.flags.DEFINE_integer('train_batch_size', 80, '''Train batch size''')
tf.app.flags.DEFINE_integer('validation_batch_size', 80, '''Validation batch size, better to be
a divisor of 10000 for this task''')
tf.app.flags.DEFINE_integer('test_batch_size', 80, '''Test batch size''')

tf.app.flags.DEFINE_float('init_lr', 0.00001, '''Initial learning rate''')
tf.app.flags.DEFINE_float('lr_decay_factor', 0.1, '''How much to decay the learning rate each
time''')
tf.app.flags.DEFINE_integer('decay_step0', 40000, '''At which step to decay the learning rate''')
tf.app.flags.DEFINE_integer('decay_step1', 60000, '''At which step to decay the learning rate''')


## The following flags define hyper-parameters modifying the training network

tf.app.flags.DEFINE_integer('num_residual_blocks', 2, '''How many residual blocks do you want''')
tf.app.flags.DEFINE_float('weight_decay', 0.0002, '''scale for l2 regularization''')


## The following flags are related to data-augmentation

tf.app.flags.DEFINE_integer('padding_size', 2, '''In data augmentation, layers of zero padding on
each side of the image''')


## If you want to load a checkpoint and continue training

tf.app.flags.DEFINE_string('ckpt_path', 'logs_cremi_adam_exp_decay_600000_steps/model.ckpt-350000', '''Checkpoint
directory to restore''')
tf.app.flags.DEFINE_boolean('is_use_ckpt', False, '''Whether to load a checkpoint and continue
training''')

tf.app.flags.DEFINE_string('test_ckpt_path', 'logs_cremi_adam_exp_decay_600000_steps/model.ckpt-350000', '''Checkpoint
directory to restore''')


train_dir = 'logs_' + FLAGS.version + '/'

learning_rate_schedule = {
    0: 0.00001, # low initial learning rate as described in paper
    20: 0.000001,
    200: 0.0000001,
    300: 0.00000001
    }