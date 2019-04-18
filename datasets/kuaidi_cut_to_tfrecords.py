#encoding=utf-8
import numpy as np;
import xml.etree.ElementTree as ET  
import tensorflow as tf
import util
from dataset_utils import int64_feature, float_feature, bytes_feature, convert_to_example
        
# input image is cropped/cut to width=2048, height=512

def cvt_to_tfrecords(output_path , data_path, gt_path):
    image_names = util.io.ls(data_path, '.jpg')#[0:10];
    print("%d images found in %s"%(len(image_names), data_path))

    with tf.python_io.TFRecordWriter(output_path) as tfrecord_writer:
        for idx, image_name in enumerate(image_names):
            oriented_bboxes = [];
            bboxes = []
            labels = [];
            labels_text = [];
            ignored = []
            path = util.io.join_path(data_path, image_name);
            print("\tconverting image: %d/%d %s"%(idx, len(image_names), image_name))
            print("path is {}".format(path))
            image_data = tf.gfile.GFile(path, 'rb')
            print("image data is {}".format(image_data))
            image_data = image_data.read()
            
            image = util.img.imread(path, rgb = True);
            shape = image.shape
            h, w = shape[0:2];
            print("height is {}, width is {}".format(h, w))
            h *= 1.0;
            w *= 1.0;
            image_name = util.str.split(image_name, '.')[0];

            gt_name = image_name + '.xml';
            gt_filepath = util.io.join_path(gt_path, gt_name);
            tree = ET.parse(gt_filepath)  
            root = tree.getroot()
            elements = root.getchildren()

            objects = [i for i in elements if i.tag == 'object']
            flag = False
            for obj in objects:
                name = [i for i in obj if i.tag == 'name'][0].text
                bndbox = [i for i in obj if i.tag == 'bndbox'][0]
                cords_4 = dict((i.tag, i.text) for i in bndbox)
                #cords_8 = [cords_4["xmin"], cords_4["ymin"], cords_4["xmax"], cords_4["ymin"], 
                #           cords_4["xmin"], cords_4["ymax"], cords_4["xmax"], cords_4["ymax"]]
                cords_8 = [cords_4["xmin"], cords_4["ymin"], cords_4["xmax"], cords_4["ymin"], 
                           cords_4["xmax"], cords_4["ymax"], cords_4["xmin"], cords_4["ymax"]]
                oriented_box = [int(i) for i in cords_8]
                oriented_box = np.asarray(oriented_box) / ([w, h] * 4);
                oriented_bboxes.append(oriented_box);
                
                xs = oriented_box.reshape(4, 2)[:, 0]                
                ys = oriented_box.reshape(4, 2)[:, 1]
                xmin = xs.min()
                xmax = xs.max()
                ymin = ys.min()
                ymax = ys.max()
                bboxes.append([xmin, ymin, xmax, ymax])
                if any([i < 0  or i > 1 for i in [xmin, ymin, xmax, ymax]]) :
                    flag = True
                # might be wrong here, but it doesn't matter because the label is not going to be used in detection
                ignored.append(0)
                labels_text.append(name); 
                labels.append(1);
            if flag:
                continue

 

            '''
            lines = util.io.read_lines(gt_filepath);
                
            for line in lines:
                line = util.str.remove_all(line, '\xef\xbb\xbf')
                gt = util.str.split(line, ',');
                oriented_box = [int(gt[i]) for i in range(8)];
                oriented_box = np.asarray(oriented_box) / ([w, h] * 4);
                oriented_bboxes.append(oriented_box);
                
                xs = oriented_box.reshape(4, 2)[:, 0]                
                ys = oriented_box.reshape(4, 2)[:, 1]
                xmin = xs.min()
                xmax = xs.max()
                ymin = ys.min()
                ymax = ys.max()
                bboxes.append([xmin, ymin, xmax, ymax])
                ignored.append(util.str.contains(gt[-1], '###'));

                # might be wrong here, but it doesn't matter because the label is not going to be used in detection
                labels_text.append(gt[-1]); 
                labels.append(1);
            '''

            example = convert_to_example(image_data, image_name, labels, ignored, labels_text, bboxes, oriented_bboxes, shape)
            tfrecord_writer.write(example.SerializeToString())
        
if __name__ == "__main__":
    root_dir = util.io.get_absolute_path('/home/kxu/workspace/data/kuaidi/2018_11_13_kuaidi_cut/label_modified')
    output_dir = util.io.get_absolute_path('~/dataset/SSD-tf/kuaidi_cut_label_modified/')
    util.io.mkdir(output_dir);

    training_data_dir = util.io.join_path(root_dir, 'images')
    training_gt_dir = util.io.join_path(root_dir,'annotations')
    cvt_to_tfrecords(output_path = util.io.join_path(output_dir, 'kuaidi_cut_label_modified.tfrecord'), data_path = training_data_dir, gt_path = training_gt_dir)

#     test_data_dir = util.io.join_path(root_dir, 'ch4_test_images')
#     test_gt_dir = util.io.join_path(root_dir,'ch4_test_localization_transcription_gt')
#     cvt_to_tfrecords(output_path = util.io.join_path(output_dir, 'icdar2015_test.tfrecord'), data_path = test_data_dir, gt_path = test_gt_dir)
