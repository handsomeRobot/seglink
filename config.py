from __future__ import print_function
from pprint import pprint
import numpy as np
from tensorflow.contrib.slim.python.slim.data import parallel_reader
import tensorflow as tf
slim = tf.contrib.slim
import util
from flags import FLAGS

global feat_shapes

global default_anchors
global defalt_anchor_map
global default_anchor_center_set
global num_anchors
global num_links

global batch_size_per_gpu
global gpus
global num_clones
global clone_scopes

# =========================================================================#
# basic config values
# =========================================================================#
gpu_device = "0"
anchor_offset = 0.5    
anchor_scale_gamma = 1.5
feat_layers = ['conv4_3','fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2']
max_height_ratio = 2
allow_soft_placement = True

# the weight applied to loss of cx, cy, w, h, theta
prior_scaling = [0.1, 0.2, 0.1, 0.2, 20]

max_neg_pos_ratio = 6
data_format = 'NHWC'

# height, width
image_shape = (512, 512) 

batch_size = 1
weight_decay = 0.0005 
num_gpus = 1 

# if < 0: all_growth = True
gpu_memory_fraction = -1 

train_with_ignored = False
seg_loc_loss_weight = 1.0
link_cls_loss_weight = 1.0
seg_conf_threshold = 0.5
link_conf_threshold = 0.5

# max number of train steps
max_number_of_steps = 1000000 

log_every_n_steps = 1 
ignore_missing_vars = True
checkpoint_exclude_scopes = None
learning_rate = 0.001

# the momentum for the MomentumOptimizer
momentum = 0.9 

using_moving_average = False
moving_average_decay = 0.9999
model_name = 'seglink_vgg'

# the number of parallel readers read data from dataset
num_readers = 1 

# the num of threads used to create the batches
num_preprocessing_threads = 1 

# update the basic config values
if FLAGS.config_file is not None:
    exec(open(FLAGS.config_file).read())

# ==============================================================================#
# Helper functions
# ==============================================================================#
def _build_anchor_map():
    global default_anchor_map
    global default_anchor_center_set
    import collections
    default_anchor_map = collections.defaultdict(list)
    for anchor_idx, anchor in enumerate(default_anchors):
        default_anchor_map[(int(anchor[1]), int(anchor[0]))].append(anchor_idx)
    default_anchor_center_set = set(default_anchor_map.keys())

    
def init_config():
    '''
    Calculate advanced config values based on basic config values
    '''
    global feat_shapes
    global default_anchors
    global num_anchors
    global num_links
    global gpus
    global num_clones
    global clone_scopes
    global batch_size_per_gpu

    # calculate the advanced config values
    from nets import anchor_layer
    from nets import seglink_symbol
    h, w = image_shape
    fake_image = tf.ones((1, h, w, 3))
    fake_net = seglink_symbol.SegLinkNet(inputs = fake_image, weight_decay = weight_decay)
    feat_shapes = fake_net.get_shapes();
    
    default_anchors, _ = anchor_layer.generate_anchors()
   
    num_anchors = len(default_anchors)
    
    _build_anchor_map()
    
    num_links = num_anchors * 8 + (num_anchors - np.prod(feat_shapes[feat_layers[0]])) * 4
    
    gpus = util.tf.get_available_gpus(num_gpus)
    
    num_clones = len(gpus)
    
    clone_scopes = ['clone_%d'%(idx) for idx in range(num_clones)]
    
    batch_size_per_gpu = batch_size / num_clones

    if batch_size_per_gpu < 1:
        raise ValueError('Invalid batch_size [=%d], resulting in 0 images per gpu.'%(batch_size))

    #print_config(print_to_file=False)
    
    return default_anchors

    
def print_config(flags=None, dataset=None, save_dir = None, print_to_file = True):
    def do_print(stream=None):
        print('\n# =========================================================================== #', file=stream)
        print('# Training flags:', file=stream)
        print('# =========================================================================== #', file=stream)
        if flags is not None:
            pprint(flags.__flags, stream=stream)

        print('\n# =========================================================================== #', file=stream)
        print('# seglink net parameters:', file=stream)
        print('# =========================================================================== #', file=stream)
        vars = globals()
        for key in vars:
            var = vars[key]
            if util.dtype.is_number(var) or util.dtype.is_str(var) or util.dtype.is_list(var) or util.dtype.is_tuple(var):
                pprint('%s=%s'%(key, str(var)), stream = stream)
            
        print('\n# =========================================================================== #', file=stream)
        print('# Training | Evaluation dataset files:', file=stream)
        print('# =========================================================================== #', file=stream)
        if dataset is not None:
            data_files = parallel_reader.get_data_files(dataset.data_sources)
            pprint(sorted(data_files), stream=stream)
            print('', file=stream)
    do_print(None)
    
    if print_to_file:
        # Save to a text file as well.
        if save_dir is None:
            save_dir = flags.train_dir
            print("save_dir is {}".format(save_dir))
            
        util.io.mkdir(save_dir)
        path = util.io.join_path(save_dir, 'training_config.txt')
        with open(path, "a") as out:
            do_print(out)

    return default_anchors 

