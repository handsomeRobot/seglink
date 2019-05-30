import os
import tensorflow as tf
from tensorflow.python.client import session
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.tools import freeze_graph

from preprocessing import ssd_vgg_preprocessing
from nets import seglink_symbol, anchor_layer
from tf_extended import seglink
from flags import FLAGS

import config

def main(_):
    config.init_config()
    
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        image = tf.placeholder(dtype=tf.int32, shape = [None, None, 3], name='input_image')
        processed_image, _, _, _, _ = ssd_vgg_preprocessing.preprocess_image(image, None, None, None, None, 
                                                   out_shape = config.image_shape,
                                                   data_format = config.data_format, 
                                                   is_training = False)
        b_image = tf.expand_dims(processed_image, axis = 0)
        net = seglink_symbol.SegLinkNet(inputs = b_image, data_format = config.data_format)
        '''
        bboxes_pred = seglink.tf_seglink_to_bbox(net.seg_scores, net.link_scores, 
                                                 net.seg_offsets, 
                                                 image_shape = b_shape, 
                                                 seg_conf_threshold = config.seg_conf_threshold,
                                                 link_conf_threshold = config.link_conf_threshold)
        '''

    predictions = {'seg_scores': net.seg_scores, 'link_scores': net.link_scores, 
                   'seg_offsets': net.seg_offsets}
    for key, value in predictions.items():
        predictions[key] = tf.identity(value, name=key)

    saver = tf.train.Saver()
    input_saver_def = saver.as_saver_def()
    output_node_names = ','.join(predictions.keys())
    tf.gfile.MakeDirs(FLAGS.output_dir)

    output_ckpt_path = os.path.join(FLAGS.output_dir, 'model.ckpt')
    inference_graph_def = tf.get_default_graph().as_graph_def()
    for node in inference_graph_def.node:
      node.device = ''
    with tf.Graph().as_default():
      tf.import_graph_def(inference_graph_def, name='')
      with session.Session() as sess:
        saver = saver_lib.Saver(saver_def=input_saver_def,
                                save_relative_paths=True)
        saver.restore(sess, FLAGS.checkpoint_path)
        saver.save(sess, output_ckpt_path)

    frozen_graph_path = os.path.join(FLAGS.output_dir,
                                     'frozen_inference_graph.pb')
    frozen_graph_def = freeze_graph.freeze_graph_with_def_protos(
        input_graph_def=tf.get_default_graph().as_graph_def(),
        input_saver_def=input_saver_def,
        input_checkpoint=FLAGS.checkpoint_path,
        output_node_names=output_node_names,
        restore_op_name='',
        filename_tensor_name='',
        output_graph=frozen_graph_path,
        clear_devices=True,
        initializer_nodes='')
    open(frozen_graph_path+'txt', 'w').write(str(frozen_graph_def))

if __name__ == '__main__':
  tf.app.run()
