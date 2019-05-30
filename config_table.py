# =========================================================================#
# basic config values
# =========================================================================#
anchor_offset = 0.5    
anchor_scale_gamma = 1.5
feat_layers = ['conv4_3','fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2']
max_height_ratio = 2

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

