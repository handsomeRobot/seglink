#encoding=utf-8
import numpy as np
import json
import xml.etree.ElementTree as ET  
import tensorflow as tf
import util
from dataset_utils import int64_feature, float_feature, bytes_feature, convert_to_example
        
# input image is cropped/cut to width=2048, height=512

def cvt_to_tfrecords(output_path , data_path, gt_path):
    image_names = util.io.ls(data_path, '.jpg')#[0:10];
    print("%d images found in %s"%(len(image_names), data_path))
    
    import_num = 0
    with tf.python_io.TFRecordWriter(output_path) as tfrecord_writer:
        for idx, image_name in enumerate(image_names):
            oriented_bboxes = [];
            bboxes = []
            labels = [];
            labels_text = [];
            ignored = []
            path = util.io.join_path(data_path, image_name);
            image_data = tf.gfile.GFile(path, 'rb')
            image_data = image_data.read()
            
            image = util.img.imread(path, rgb = True);
            shape = image.shape
            h, w = shape[0:2];
            h *= 1.0;
            w *= 1.0;
            image_name = util.str.split(image_name, '.')[0];

            gt_name = image_name + '.json';
            gt_filepath = util.io.join_path(gt_path, gt_name);
            print(">>>> gt_filepath is {}".format(gt_filepath))
            with open(gt_filepath) as f:
                gt_info = json.load(f)

            objects = gt_info.get("shapes", [])
            for obj in objects:
                name = obj.get("label", "unk")
                points = obj.get("points", [])
                oriented_box = np.asarray(obj.get("points", [])).ravel()
                oriented_box /= ([w, h] * int(len(oriented_box) / 2))
                oriented_bboxes.append(oriented_box)
                xs = oriented_box.reshape(-1, 2)[:, 0]
                ys = oriented_box.reshape(-1, 2)[:, 1]
                xmin = xs.min()
                xmax = xs.max()
                ymin = ys.min()
                ymax = ys.max()
                bboxes.append([xmin, ymin, xmax, ymax])
                # might be wrong here, but it doesn't matter because the label is not going to be used in detection
                ignored.append(0)
                labels_text.append(name); 
                labels.append(1);

            '''
            # 统一多边形的顶点数量，不足的用最后一个顶点补足
            max_polypoints_num = max([len(poly) / 2 for poly in oriented_bboxes])
            for i, poly in enumerate(oriented_bboxes):
                fill_num = int(max_polypoints_num - len(poly) / 2)
                last_vertex = poly[-2:]
                tmp = np.concatenate([poly, np.asarray(list(last_vertex) * fill_num)])
                oriented_bboxes[i] = np.concatenate([poly, np.asarray(list(last_vertex) * fill_num)])

            print(oriented_bboxes)
            '''

            try:
                example = convert_to_example(image_data, image_name, labels, ignored, labels_text, bboxes, oriented_bboxes, shape, vertex_num=4)
                tfrecord_writer.write(example.SerializeToString())
                import_num += 1
            except:
                pass
    print(import_num)
            
if __name__ == "__main__":
    #root_dir = util.io.get_absolute_path('/home/kxu/workspace/data/table_seglink_0523')
    #root_dir = util.io.get_absolute_path('/home/kxu/workspace/data/seg_table_complx_test_0523')
    root_dir = util.io.get_absolute_path('/home/kxu/workspace/data/table_seglink_poly_0528')
    output_dir = util.io.get_absolute_path('~/dataset/SSD-tf/table_extract_test_0528/')
    util.io.mkdir(output_dir);

    training_data_dir = util.io.join_path(root_dir, 'images')
    training_gt_dir = util.io.join_path(root_dir, 'annotations')
    cvt_to_tfrecords(output_path = util.io.join_path(output_dir, 'table_extract_test_0528.tfrecord'), data_path = training_data_dir, gt_path = training_gt_dir)

#     test_data_dir = util.io.join_path(root_dir, 'ch4_test_images')
#     test_gt_dir = util.io.join_path(root_dir,'ch4_test_localization_transcription_gt')
#     cvt_to_tfrecords(output_path = util.io.join_path(output_dir, 'icdar2015_test.tfrecord'), data_path = test_data_dir, gt_path = test_gt_dir)
