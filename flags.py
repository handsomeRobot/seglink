import tensorflow as tf

# =========================================================================== #
# Checkpoint and running Flags
# =========================================================================== #
tf.app.flags.DEFINE_string('train_dir', None, 
                           'the path to store checkpoints and eventfiles for summaries')

tf.app.flags.DEFINE_string('checkpoint_path', None, 
   'the path of pretrained model to be used. If there are checkpoints in train_dir, this config will be ignored.')

tf.app.flags.DEFINE_string('config_file', None, 'config_file')

# =========================================================================== # 
# Dataset Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'dataset_name', None, 'The name of the dataset to load.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')
tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

# =========================================================================== #
# Inference graph export flags
# =========================================================================== #
tf.app.flags.DEFINE_string("output_dir", None, "The directory for the output graph files.")
tf.app.flags.DEFINE_string("checkpoint", None, "The directory of the original checkpoints.")

FLAGS = tf.app.flags.FLAGS


