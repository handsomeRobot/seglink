#test code to make sure the ground truth calculation and data batch works well.

import numpy as np
import tensorflow as tf # test
from tensorflow.python.ops import control_flow_ops

from datasets import dataset_factory
from preprocessing import ssd_vgg_preprocessing
from tf_extended import seglink
import util
import cv2
from nets import seglink_symbol, anchor_layer
from preprocessing.ssd_vgg_preprocessing import tf_summary_image


slim = tf.contrib.slim
import config
# =========================================================================== #
# Checkpoint and running Flags
# =========================================================================== #
tf.app.flags.DEFINE_bool('train_with_ignored', False, 
                           'whether to use ignored bbox (in ic15) in training.')
tf.app.flags.DEFINE_float('seg_loc_loss_weight', 1.0, 'the loss weight of segment localization')
tf.app.flags.DEFINE_float('link_cls_loss_weight', 1.0, 'the loss weight of linkage classification loss')

tf.app.flags.DEFINE_string('train_dir', None, 
                           'the path to store checkpoints and eventfiles for summaries')

tf.app.flags.DEFINE_string('checkpoint_path', None, 
   'the path of pretrained model to be used. If there are checkpoints in train_dir, this config will be ignored.')

tf.app.flags.DEFINE_float('gpu_memory_fraction', -1, 
                          'the gpu memory fraction to be used. If less than 0, allow_growth = True is used.')

tf.app.flags.DEFINE_integer('batch_size', None, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer('num_gpus', 1, 'The number of gpus can be used.')
tf.app.flags.DEFINE_integer('max_number_of_steps', 1000000, 'The maximum number of training steps.')
tf.app.flags.DEFINE_integer('log_every_n_steps', 1, 'log frequency')
tf.app.flags.DEFINE_bool("ignore_missing_vars", True, '')
tf.app.flags.DEFINE_string('checkpoint_exclude_scopes', None, 'checkpoint_exclude_scopes')

# =========================================================================== #
# Optimizer configs.
# =========================================================================== #
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning rate.')
tf.app.flags.DEFINE_float('momentum', 0.9, 'The momentum for the MomentumOptimizer')
tf.app.flags.DEFINE_float('weight_decay', 0.0005, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_bool('using_moving_average', False, 'Whether to use ExponentionalMovingAverage')
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999, 'The decay rate of ExponentionalMovingAverage')

# =========================================================================== #
# I/O and preprocessing Flags.
# =========================================================================== #
tf.app.flags.DEFINE_integer(
    'num_readers', 1,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 1,
    'The number of threads used to create the batches.')

# =========================================================================== #
# Dataset Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'dataset_name', None, 'The name of the dataset to load.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')
tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_string(
    'model_name', 'seglink_vgg', 'The name of the architecture to train.')
tf.app.flags.DEFINE_integer('train_image_width', 512, 'Train image size')
tf.app.flags.DEFINE_integer('train_image_height', 512, 'Train image size')


FLAGS = tf.app.flags.FLAGS

def config_initialization():
    # image shape and feature layers shape inference
    image_shape = (FLAGS.train_image_height, FLAGS.train_image_width)
    
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')
    tf.logging.set_verbosity(tf.logging.DEBUG)
    util.init_logger(log_file = 'log_train_seglink_%d_%d.log'%image_shape, log_path = FLAGS.train_dir, stdout = False, mode = 'a')
    
    
    default_anchors = config.init_config(image_shape, 
                       batch_size = int(FLAGS.batch_size), 
                       weight_decay = FLAGS.weight_decay, 
                       num_gpus = FLAGS.num_gpus, 
                       train_with_ignored = FLAGS.train_with_ignored,
                       seg_loc_loss_weight = FLAGS.seg_loc_loss_weight, 
                       link_cls_loss_weight = FLAGS.link_cls_loss_weight, 
                       )

    batch_size = int(config.batch_size)
    batch_size_per_gpu = config.batch_size_per_gpu
        
    tf.summary.scalar('batch_size', batch_size)
    tf.summary.scalar('batch_size_per_gpu', batch_size_per_gpu)

    util.proc.set_proc_name(FLAGS.model_name + '_' + FLAGS.dataset_name)
    
    dataset = dataset_factory.get_dataset(FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)
    config.print_config(FLAGS, dataset)
    #print("default_anchors is {}".format(default_anchors))
    return dataset, default_anchors

def create_dataset_batch_queue(dataset, default_anchors):
    with tf.device('/cpu:0'):
        with tf.name_scope(FLAGS.dataset_name + '_data_provider'):
            provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                num_readers=FLAGS.num_readers,
                common_queue_capacity=50 * int(config.batch_size),
                common_queue_min=30 * int(config.batch_size),
                shuffle=True)
        # Get for SSD network: image, labels, bboxes.
        [image, gignored, gbboxes, x1, x2, x3, x4, y1, y2, y3, y4] = provider.get([
                                                         'image',
                                                         'object/ignored',
                                                         'object/bbox', 
                                                         'object/oriented_bbox/x1',
                                                         'object/oriented_bbox/x2',
                                                         'object/oriented_bbox/x3',
                                                         'object/oriented_bbox/x4',
                                                         'object/oriented_bbox/y1',
                                                         'object/oriented_bbox/y2',
                                                         'object/oriented_bbox/y3',
                                                         'object/oriented_bbox/y4',
                                                         ])
        gxs = tf.transpose(tf.stack([x1, x2, x3, x4])) #shape = (N, 4)
        gys = tf.transpose(tf.stack([y1, y2, y3, y4]))
        image = tf.identity(image, 'input_image')
        
        # Pre-processing image, labels and bboxes.
        image, gignored, gbboxes, gxs, gys = ssd_vgg_preprocessing.preprocess_image(image, gignored, gbboxes, gxs, gys,
                                                           out_shape = config.image_shape,
                                                           data_format = config.data_format, 
                                                           is_training = True,
                                                           default_anchors = default_anchors)
        image = tf.identity(image, 'processed_image')
        
        def seg_loc_to_bbox(x, image):
            height, width = image.shape[:2]
            loc = x[:, :4]
            bbox = loc.copy()
            for i in range(len(bbox)):
                #bbox[i] = [(loc[i][0] - 0.5 * loc[i][2])/width, 
                #           (loc[i][1] - 0.5 * loc[i][3])/height, 
                #           (loc[i][0] + 0.5 * loc[i][2])/width, 
                #           (loc[i][1] + 0.5 * loc[i][3])/height]
                bbox[i] = [(loc[i][1] - 0.5 * loc[i][3])/height, 
                           (loc[i][0] - 0.5 * loc[i][2])/width,
                           (loc[i][1] + 0.5 * loc[i][3])/height,
                           (loc[i][0] + 0.5 * loc[i][2])/width] 
            bbox = np.asarray(bbox)
            return bbox


        def bboxes_to_xys(bboxes, image_shape):
            """Convert Seglink bboxes to xys, i.e., eight points
            The `image_shape` is used to to make sure all points return are valid, i.e., within image area
            """
            if len(bboxes) == 0:
                assert np.ndim(bboxes) == 2 and np.shape(bboxes)[-1] == 5, 'invalid `bboxes` param with shape =  ' + str(np.shape(bboxes))
            h, w = image_shape[0:2]
           
            xys = np.zeros((len(bboxes), 8))
            for bbox_idx, bbox in enumerate(bboxes):
                bbox = ((bbox[0], bbox[1]), (bbox[2], bbox[3]), bbox[4])
                points = cv2.boxPoints(bbox)
                points = np.int0(points)
                points = np.reshape(points, -1)
                xys[bbox_idx, :] = points
            return xys


        def draw_oriented_bbox(bboxes, image):
            bboxes = bboxes_to_xys(bboxes, image.shape)
            for bbox in bboxes:
                if len(bbox) == 0:
                    continue
                points = [int(v) for v in bbox[0:8]]
                points = np.reshape(points, (4, 2))
                cnts = util.img.points_to_contours(points)
                image = util.img.draw_contours(image, cnts, -1, color = [0, 0, 255], border_width = 1)
            return image
       
 
        # calculate ground truth
        seg_label, seg_loc, link_label, seg_locations, bbox_mask, rects  = seglink.tf_get_all_seglink_gt(gxs, gys, gignored, image)
        # summary bbox_mask for debug
        bbox_mask = tf.expand_dims(bbox_mask, -1)
        bbox_mask = tf.expand_dims(bbox_mask, 0)
        tmp = image + tf.cast(bbox_mask, tf.float32) * 50
        tf.summary.image("bbox_mask", tf.cast(tmp, tf.float32))
        # summary seg_bbox loc for debug
        #seg_bbox = tf.py_func(seg_loc_to_bbox, [seg_locations, image], tf.float32)
        #seg_bbox.set_shape(seg_locations.get_shape())
        #tf_summary_image(image, seg_bbox, name="seg_locations")
        # summary rects for debug
        tmp = tf.py_func(draw_oriented_bbox, [seg_locations, image], tf.float32)
        tmp = tf.expand_dims(tmp, 0)
        tf.summary.image("rects", tmp)

        # batch them
        b_image, b_seg_label, b_seg_loc, b_link_label = tf.train.batch(
            [image, seg_label, seg_loc, link_label],
            batch_size = int(config.batch_size_per_gpu),
            num_threads= FLAGS.num_preprocessing_threads,
            capacity = 50)

        batch_queue = slim.prefetch_queue.prefetch_queue(
            [b_image, b_seg_label, b_seg_loc, b_link_label],
            capacity = 50) 
    return batch_queue    

def sum_gradients(clone_grads):                        
    averaged_grads = []
    for grad_and_vars in zip(*clone_grads):
        grads = []
        var = grad_and_vars[0][1]
        for g, v in grad_and_vars:
            assert v == var
            grads.append(g)
        grad = tf.add_n(grads, name = v.op.name + '_summed_gradients')
        averaged_grads.append((grad, v))
        
        tf.summary.histogram("variables_and_gradients_" + grad.op.name, grad)
        tf.summary.histogram("variables_and_gradients_" + v.op.name, v)
        tf.summary.scalar("variables_and_gradients_" + grad.op.name+'_mean/var_mean', tf.reduce_mean(grad)/tf.reduce_mean(var))
        tf.summary.scalar("variables_and_gradients_" + v.op.name+'_mean', tf.reduce_mean(var))
    return averaged_grads


def create_clones(batch_queue):        
    with tf.device('/cpu:0'):
        global_step = slim.create_global_step()
        learning_rate = tf.constant(FLAGS.learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)
        #optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=FLAGS.momentum, name='Momentum')
        #optimizer = tf.train.AdamOptimizer(learning_rate, name='Adam')
        optimizer = tf.train.GradientDescentOptimizer(learning_rate, name='SGD')
        
    # place clones
    seglink_loss = 0; # for summary only
    gradients = []
    for clone_idx, gpu in enumerate(config.gpus):
        #do_summary = clone_idx == 0 # only summary on the first clone
        do_summary = True
        with tf.variable_scope(tf.get_variable_scope(), reuse = True):# the variables has been created in config.init_config
            with tf.name_scope(config.clone_scopes[clone_idx]) as clone_scope:
                with tf.device(gpu) as clone_device:
                    b_image, b_seg_label, b_seg_loc, b_link_label = batch_queue.dequeue()
                    net = seglink_symbol.SegLinkNet(inputs = b_image, data_format = config.data_format)
                    '''
                    def seg_loc_to_bbox(x):
                        height, width = config.image_shape
                        loc = x[:, :4]
                        bbox = loc.copy()
                        for i in range(len(bbox)):
                            bbox[i] = [(loc[i][1] - 0.5 * loc[i][3])/height, 
                                       (loc[i][0] - 0.5 * loc[i][2])/width,
                                       (loc[i][1] + 0.5 * loc[i][3])/height,
                                       (loc[i][0] + 0.5 * loc[i][2])/width] 
                        bbox = np.asarray(bbox)
                        return bbox
                    seg_locs = tf.unstack(b_seg_loc)
                    images = tf.unstack(b_image)
                    print("seg_locs is {}".format(seg_locs))
                    print("images is {}".format(images))
                    for image, seg_loc in zip(images, seg_locs):
                        bbox = tf.py_func(seg_loc_to_bbox, [seg_loc], tf.float32)
                        bbox.set_shape((seg_loc.get_shape()[0], 4))
                        tf_summary_image(image, bbox, "predicted")
                    '''
                    
                    # build seglink loss
                    net.build_loss(seg_labels = b_seg_label, 
                                   seg_offsets = b_seg_loc, 
                                   link_labels = b_link_label,
                                   do_summary = do_summary)
                    
                    
                    # gather seglink losses
                    losses = tf.get_collection(tf.GraphKeys.LOSSES, clone_scope)
                    assert len(losses) ==  3  # 3 is the number of seglink losses: seg_cls, seg_loc, link_cls
                    total_clone_loss = tf.add_n(losses) / config.num_clones
                    seglink_loss = seglink_loss + total_clone_loss

                    # gather regularization loss and add to clone_0 only
                    if clone_idx == 0:
                        regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                        total_clone_loss = total_clone_loss + regularization_loss
                    
                    # compute clone gradients
                    clone_gradients = optimizer.compute_gradients(total_clone_loss)# all variables will be updated.
                    gradients.append(clone_gradients)
                    
    tf.summary.scalar('seglink_loss', seglink_loss)
    tf.summary.scalar('regularization_loss', regularization_loss)
    
    # add all gradients together
    # note that the gradients do not need to be averaged, because the average operation has been done on loss.
    averaged_gradients = sum_gradients(gradients)
    
    update_op = optimizer.apply_gradients(averaged_gradients, global_step=global_step)
    
    train_ops = [update_op]
    
    # moving average
    if FLAGS.using_moving_average:
        tf.logging.info('using moving average in training, \
        with decay = %f'%(FLAGS.moving_average_decay))
        ema = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay)
        ema_op = ema.apply(tf.trainable_variables())
        with tf.control_dependencies([update_op]):# ema after updating
            train_ops.append(tf.group(ema_op))
            
    train_op = control_flow_ops.with_dependencies(train_ops, seglink_loss, name='train_op')
    return train_op

    
    
def train(train_op):
    summary_op = tf.summary.merge_all()
    sess_config = tf.ConfigProto(log_device_placement = False, allow_soft_placement = True)
    if FLAGS.gpu_memory_fraction < 0:
        sess_config.gpu_options.allow_growth = True
    elif FLAGS.gpu_memory_fraction > 0:
        sess_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction;
    
    init_fn = util.tf.get_init_fn(checkpoint_path = FLAGS.checkpoint_path, train_dir = FLAGS.train_dir, 
                          ignore_missing_vars = FLAGS.ignore_missing_vars, checkpoint_exclude_scopes = FLAGS.checkpoint_exclude_scopes)
    saver = tf.train.Saver(max_to_keep = 500, write_version = 2)
    slim.learning.train(
            train_op,
            logdir = FLAGS.train_dir,
            init_fn = init_fn,
            summary_op = summary_op,
            number_of_steps = FLAGS.max_number_of_steps,
            log_every_n_steps = FLAGS.log_every_n_steps,
            save_summaries_secs = 60,
            saver = saver,
            save_interval_secs = 1200,
            session_config = sess_config
    )


def main(_):
    # The choice of return dataset object via initialization method maybe confusing, 
    # but I need to print all configurations in this method, including dataset information. 
    dataset, default_anchors = config_initialization()   
    
    batch_queue = create_dataset_batch_queue(dataset, default_anchors)
    train_op = create_clones(batch_queue)
    train(train_op)
    
    
if __name__ == '__main__':
    tf.app.run()
