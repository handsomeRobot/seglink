# Copyright 2015 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Pre-processing images for SSD-type networks.
"""
from enum import Enum, IntEnum
import numpy as np
import random
import math

import tensorflow as tf
import tf_extended as tfe

from tensorflow.python.ops import control_flow_ops

from preprocessing import tf_image
import cv2

slim = tf.contrib.slim

# Resizing strategies.
Resize = IntEnum('Resize', ('NONE',                # Nothing!
                            'CENTRAL_CROP',        # Crop (and pad if necessary).
                            'PAD_AND_RESIZE',      # Pad, and resize to output shape.
                            'WARP_RESIZE'))        # Warp resize.

# VGG mean parameters.
_R_MEAN = 123.
_G_MEAN = 117.
_B_MEAN = 104.

# Some training pre-processing parameters.
BBOX_CROP_OVERLAP = 0.1         # Minimum overlap to keep a bbox after cropping.
MIN_OBJECT_COVERED = 0.5
CROP_ASPECT_RATIO_RANGE = (0.5, 2.)  # Distortion ratio during cropping.
EVAL_SIZE = (300, 300)
AREA_RANGE = [0.1, 1]
FLIP = False

def cv2_draw_polygon(image, xs, ys):
    points = np.dstack([xs, ys]) # shape (N, 4, 2)
    points = [np.int0(p) for p in points]
    image = cv2.polylines(image.copy() * 255, points, True, [0, 0, 255], 3)
    return image


def _rotate_image_with_bounding_box(image, bbox_lst, xs, ys):
    '''
    Rotate the image and bounding box to an angle between [-180, 180].
    '''
    angle = random.randint(-45, 45)
    print(">>>>>>>> angle is {}".format(angle))
    angle_rad = 3.1415926 * angle / 180 
    
    # get the rotation matrix and rotate the image
    height, width = image.shape[:2]
    M = cv2.getRotationMatrix2D((0.5*width, 0.5*height), angle, 1.)
    rotated_image = cv2.warpAffine(image, M)

    # update the coordinates
    xs = xs * width
    ys = ys * height
    new_xs = M[0, 0] * xs + M[0, 1] * ys + M[0, 2] 
    new_ys = M[1, 0] * xs + M[1, 1] * ys + M[1, 2] 
   
    # draw polygons for debuging
    #debug_image = tf.py_func(cv2_draw_polygon, [rotated_img, new_xs, new_ys], tf.float32)
    #debug_image = tf.expand_dims(debug_image, 0)
    #tf.summary.image("rotated image and bboxes", debug_image)

    # scale to 0-1
    new_xs = new_xs * 1.0 / width
    new_ys = new_ys * 1.0 / height

    # get the new bboxes 
    xmin = tf.reduce_min(new_xs, axis=[1])
    ymin = tf.reduce_min(new_ys, axis=[1])
    xmax = tf.reduce_max(new_xs, axis=[1])
    ymax = tf.reduce_max(new_ys, axis=[1])
    bboxes = tf.stack([ymin, xmin, ymax, xmax], axis=1) 
    #'''

    #return rotated_img, bbox_lst, xs, ys
    return rotated_img, bboxes, new_xs, new_ys



def rotate_image_with_bounding_box(image, bbox_lst, xs, ys):
    '''
    Rotate the image and bounding box to an angle between [-180, 180].
    '''

    def _rotate_image(image, rotate_angle, xs, ys):
        height, width = image.shape[:2]
        M = cv2.getRotationMatrix2D((0.5 * width, 0.5 * height), rotate_angle[0], 1.)
        rotated_image = cv2.warpAffine(image, M, (width, height)) 
        # update the coordinates
        xs = xs * width
        ys = ys * height
        new_xs = M[0, 0] * xs + M[0, 1] * ys + M[0, 2] 
        new_ys = M[1, 0] * xs + M[1, 1] * ys + M[1, 2] 
        # debug image
        debug_image = cv2_draw_polygon(rotated_image, new_xs, new_ys)
        # scale to 0-1
        new_xs = new_xs * 1.0 / width
        new_ys = new_ys * 1.0 / height
        # get the new bboxes
        xmin = np.min(new_xs, axis=1).reshape([-1, 1])
        ymin = np.min(new_ys, axis=1).reshape([-1, 1])
        xmax = np.max(new_xs, axis=1).reshape([-1, 1])
        ymax = np.max(new_ys, axis=1).reshape([-1, 1])
        bboxes = np.concatenate([ymin, xmin, ymax, xmax], axis=1) 

        return rotated_image, debug_image, new_xs, new_ys, bboxes


    angle = tf.random_uniform([1], -45, 45)
    rotated_image, debug_image, new_xs, new_ys, new_bboxes = tf.py_func(_rotate_image, 
            [image, angle, xs, ys], [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])

    rotated_image.set_shape(image.get_shape().as_list())
    debug_image.set_shape(image.get_shape().as_list())
    new_xs.set_shape(xs.get_shape().as_list())
    new_ys.set_shape(ys.get_shape().as_list())
    new_bboxes.set_shape(bbox_lst.get_shape().as_list())

    # draw polygons for debuging
    debug_image = tf.expand_dims(debug_image, 0)
    tf.summary.image("rotated image and bboxes", debug_image)

    #return rotated_img, bbox_lst, xs, ys
    return rotated_image, new_bboxes, new_xs, new_ys


def tf_image_whitened(image, means=[_R_MEAN, _G_MEAN, _B_MEAN]):
    """Subtracts the given means from each image channel.

    Returns:
        the centered image.
    """
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    mean = tf.constant(means, dtype=image.dtype)
    image = image - mean
    return image


def tf_image_unwhitened(image, means=[_R_MEAN, _G_MEAN, _B_MEAN], to_int=True):
    """Re-convert to original image distribution, and convert to int if
    necessary.

    Returns:
      Centered image.
    """
    mean = tf.constant(means, dtype=image.dtype)
    image = image + mean
    if to_int:
        image = tf.cast(image, tf.int32)
    return image


def np_image_unwhitened(image, means=[_R_MEAN, _G_MEAN, _B_MEAN], to_int=True):
    """Re-convert to original image distribution, and convert to int if
    necessary. Numpy version.

    Returns:
      Centered image.
    """
    img = np.copy(image)
    img += np.array(means, dtype=img.dtype)
    if to_int:
        img = img.astype(np.uint8)
    return img


def tf_summary_image(image, bboxes, name='image', unwhitened=False):
    """Add image with bounding boxes to summary.
    """
    if unwhitened:
        image = tf_image_unwhitened(image)
    image = tf.expand_dims(image, 0)
    bboxes = tf.expand_dims(bboxes, 0)
    image_with_box = tf.image.draw_bounding_boxes(image, bboxes)
    tf.summary.image(name, image_with_box)


def apply_with_random_selector(x, func, num_cases):
    """Computes func(x, sel), with sel sampled from [0...num_cases-1].

    Args:
        x: input Tensor.
        func: Python function to apply.
        num_cases: Python int32, number of cases to sample sel from.

    Returns:
        The result of func(x, sel), where func receives the value of the
        selector as a python integer, but sel is sampled dynamically.
    """
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    # Pass the real x only to one of the func calls.
    return control_flow_ops.merge([
            func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
            for case in range(num_cases)])[0]


def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
    """Distort the color of a Tensor image.

    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.

    Args:
        image: 3-D Tensor containing single image in [0, 1].
        color_ordering: Python int, a type of distortion (valid values: 0-3).
        fast_mode: Avoids slower ops (random_hue and random_contrast)
        scope: Optional scope for name_scope.
    Returns:
        3-D Tensor color-distorted image on range [0, 1]
    Raises:
        ValueError: if color_ordering not in [0, 3]
    """
    with tf.name_scope(scope, 'distort_color', [image]):
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
            else:
                raise ValueError('color_ordering must be in [0, 3]')
        # The random_* ops do not necessarily clamp.
        return tf.clip_by_value(image, 0.0, 1.0)


def distorted_bounding_box_crop(image,
                                labels,
                                bboxes,
                                xs, ys, 
                                min_object_covered,
                                aspect_ratio_range,
                                area_range,
                                max_attempts = 200,
                                scope=None):
    """Generates cropped_image using a one of the bboxes randomly distorted.

    See `tf.image.sample_distorted_bounding_box` for more documentation.

    Args:
        image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
        bbox: 2-D float Tensor of bounding boxes arranged [num_boxes, coords]
            where each coordinate is [0, 1) and the coordinates are arranged
            as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
            image.
        min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
            area of the image must contain at least this fraction of any bounding box
            supplied.
        aspect_ratio_range: An optional list of `floats`. The cropped area of the
            image must have an aspect ratio = width / height within this range.
        area_range: An optional list of `floats`. The cropped area of the image
            must contain a fraction of the supplied image within in this range.
        max_attempts: An optional `int`. Number of attempts at generating a cropped
            region of the image of the specified constraints. After `max_attempts`
            failures, return the entire image.
        scope: Optional scope for name_scope.
    Returns:
        A tuple, a 3-D Tensor cropped_image and the distorted bbox
    """
    with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bboxes, xs, ys]):
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].
        bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
                tf.shape(image),
                bounding_boxes=tf.expand_dims(bboxes, 0),
                min_object_covered=min_object_covered,
                aspect_ratio_range=aspect_ratio_range,
                area_range=area_range,
                max_attempts=max_attempts,
                use_image_if_no_bounding_boxes=True)

        # Draw the bounding box in an image summary.
        image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0), distort_bbox)
        tf.summary.image('images_with_box', image_with_box)

        distort_bbox = distort_bbox[0, 0]

        # Crop the image to the specified bounding box.
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        # Restore the shape since the dynamic slice loses 3rd dimension.
        cropped_image.set_shape([None, None, 3])

        # Update bounding boxes: resize and filter out.
        bboxes, xs, ys = tfe.bboxes_resize(distort_bbox, bboxes, xs, ys)
        labels, bboxes, xs, ys = tfe.bboxes_filter_overlap(labels, bboxes, xs, ys, 
                                                threshold=BBOX_CROP_OVERLAP, assign_negative = False)
        return cropped_image, labels, bboxes, xs, ys, distort_bbox


def preprocess_for_train(image, labels, bboxes, xs, ys,
                         out_shape, data_format='NHWC',
                         scope='ssd_preprocessing_train',
                         default_anchors=None):
    """Preprocesses the given image for training.

    Note that the actual resizing scale is sampled from
        [`resize_size_min`, `resize_size_max`].

    Args:
        image: A `Tensor` representing an image of arbitrary size.
        output_height: The height of the image after preprocessing.
        output_width: The width of the image after preprocessing.
        resize_side_min: The lower bound for the smallest side of the image for
            aspect-preserving resizing.
        resize_side_max: The upper bound for the smallest side of the image for
            aspect-preserving resizing.

    Returns:
        A preprocessed image.
    """
    fast_mode = False
    with tf.name_scope(scope, 'ssd_preprocessing_train', [image, labels, bboxes]):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')
        # Convert to float scaled [0, 1].
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # tf_summary_image(image, bboxes, 'image_with_bboxes')

        # Distort image and bounding boxes.
        #'''
        dst_image, labels, bboxes, xs, ys, distort_bbox = \
            distorted_bounding_box_crop(image, labels, bboxes, xs, ys,
                                        min_object_covered = MIN_OBJECT_COVERED,
                                        aspect_ratio_range = CROP_ASPECT_RATIO_RANGE, 
                                        area_range = AREA_RANGE)
        #'''

        # Resize image to output size.
        dst_image = tf_image.resize_image(dst_image, out_shape,
                                          method=tf.image.ResizeMethod.BILINEAR,
                                          align_corners=False)
        

        # rotate the image (and the bounding box) to a random angle between [-15, 15]
        #print(">>>>>> resize image is {}".format(dst_image))
        #dst_image = tf.py_func(draw_pic, [dst_image], tf.float32)
        #print(">>>>>> after py_func is {}".format(dst_image))
        dst_image, bboxes, xs, ys = rotate_image_with_bounding_box(dst_image, bboxes, xs, ys)
 
        tf_summary_image(dst_image, bboxes, 'image_shape_distorted')

        # draw default anchors by the way
        #print("image shape is {}".format(dst_image.get_shape().as_list()))
        height, width = dst_image.get_shape().as_list()[:2]
        flatted_anchors = default_anchors.reshape([-1, 4])
        import collections
        res = collections.defaultdict(list) 
        for anchor in flatted_anchors:
            #print("anchor is {}".format(anchor))
            cx = anchor[0]
            cy = anchor[1]
            w = anchor[2]
            h = anchor[3]
            rank = w
            x_min = (cx - 0.5 * w) / width
            y_min = (cy - 0.5 * h) / height
            x_max = (cx + 0.5 * w) / width
            y_max = (cy + 0.5 * h) / height
            res[rank].append([y_min, x_min, y_max, x_max])
        for rank, anchors in res.items():
            anchors = tf.constant(anchors, dtype=tf.float32)
            tf_summary_image(dst_image, anchors, name='anchors_' + str(rank))
 
        # Randomly flip the image horizontally.
        if FLIP:
            dst_image, bboxes = tf_image.random_flip_left_right(dst_image, bboxes)

        # Randomly distort the colors. There are 4 ways to do it.
        dst_image = apply_with_random_selector(
                dst_image,
                lambda x, ordering: distort_color(x, ordering, fast_mode),
                num_cases=4)
        tf_summary_image(dst_image, bboxes, 'image_color_distorted')

        # Rescale to VGG input scale.
        image = dst_image * 255.
        image = tf_image_whitened(image, [_R_MEAN, _G_MEAN, _B_MEAN])
        # Image data format.
        if data_format == 'NCHW':
            image = tf.transpose(image, perm=(2, 0, 1))

        return image, labels, bboxes, xs, ys


def preprocess_for_eval(image, labels, bboxes, xs, ys,
                        out_shape=EVAL_SIZE, data_format='NHWC',
                        difficults=None, resize=Resize.WARP_RESIZE,
                        scope='ssd_preprocessing_train'):
    """Preprocess an image for evaluation.

    Args:
        image: A `Tensor` representing an image of arbitrary size.
        out_shape: Output shape after pre-processing (if resize != None)
        resize: Resize strategy.

    Returns:
        A preprocessed image.
    """
    with tf.name_scope(scope):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')

        image = tf.to_float(image)
        image = tf_image_whitened(image, [_R_MEAN, _G_MEAN, _B_MEAN])

        if resize == Resize.NONE:
            pass
        else:
            image = tf_image.resize_image(image, out_shape,
                                          method=tf.image.ResizeMethod.BILINEAR,
                                          align_corners=False)

        # Image data format.
        if data_format == 'NCHW':
            image = tf.transpose(image, perm=(2, 0, 1))
        return image, labels, bboxes, xs, ys


def preprocess_image(image,
                     labels,
                     bboxes,
                     xs, ys,
                     out_shape,
                     data_format = 'NHWC',
                     is_training=False,
                     default_anchors=None,
                     **kwargs):
    """Pre-process an given image.

    Args:
      image: A `Tensor` representing an image of arbitrary size.
      output_height: The height of the image after preprocessing.
      output_width: The width of the image after preprocessing.
      is_training: `True` if we're preprocessing the image for training and
        `False` otherwise.
      resize_side_min: The lower bound for the smallest side of the image for
        aspect-preserving resizing. If `is_training` is `False`, then this value
        is used for rescaling.
      resize_side_max: The upper bound for the smallest side of the image for
        aspect-preserving resizing. If `is_training` is `False`, this value is
         ignored. Otherwise, the resize side is sampled from
         [resize_size_min, resize_size_max].

    Returns:
      A preprocessed image.
    """
    if is_training:
        return preprocess_for_train(image, labels, bboxes, xs, ys,
                                    out_shape=out_shape,
                                    data_format=data_format,
                                    default_anchors=default_anchors)
    else:
        return preprocess_for_eval(image, labels, bboxes, xs, ys,
                                   out_shape=out_shape,
                                   data_format=data_format,
                                   **kwargs)
