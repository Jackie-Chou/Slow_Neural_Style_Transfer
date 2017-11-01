import os

import argparse
import functools
import time
import vgg
import tensorflow as tf
import numpy as np
from utils import *

STYLE_LAYERS = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
CONTENT_LAYER = 'relu4_2'

parser = argparse.ArgumentParser(description="transfer image style")

parser.add_argument('-s', '--style_path', type=str, required=True)
parser.add_argument('-c', '--content_path', type=str, required=True)
parser.add_argument('-g', '--gpu_id', default="0", type=str, required=False)
parser.add_argument('-o', '--output_path', default='../data/styled/output.jpg', type=str, required=False)
parser.add_argument('-t', '--tem_save_path', default='../data/tem', type=str, required=False)
parser.add_argument('-i', '--init_img', default='content', type=str, required=False)
parser.add_argument('-m', '--max_steps', default=500, type=int, required=False)
parser.add_argument('-v', '--save_steps', default=100, type=int, required=False)
parser.add_argument('-x', '--content_weight', default=1e-3, type=float, required=False)
parser.add_argument('-y', '--style_weight', default=0.2, type=float, required=False)
parser.add_argument('-z', '--tv_weight', default=1e-5, type=float, required=False)
parser.add_argument('-r', '--learning_rate', default=1e3, type=float, required=False)


def optimize(content_target, style_target, content_weight, style_weight, tv_weight, 
             learning_rate, vgg_path, max_steps, save_steps, 
             init_img, tem_save_path, output_path):  
   
    content_shape = (1,) + content_target.shape
    style_shape = (1,) + style_target.shape
    output_shape = content_shape

    style_features = {}
    # precompute style features
    with tf.Graph().as_default(), tf.Session() as sess:
        style_image = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
        style_image_pre = vgg.preprocess(style_image)
        net = vgg.net(vgg_path, style_image_pre)
        style_pre = np.array([style_target])
        for layer in STYLE_LAYERS:
            features = net[layer].eval(feed_dict={style_image:style_pre})
            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) / features.size
            style_features[layer] = gram

    
    content_features = {}
    # precompute content features
    with tf.Graph().as_default(), tf.Session() as sess:
        content_image = tf.placeholder(tf.float32, shape=content_shape, name="content_image")
        pre_image = vgg.preprocess(content_image)
        content_pre = np.array([content_target])
        content_net = vgg.net(vgg_path, pre_image)
        content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER].eval(feed_dict={content_image: content_pre})

    with tf.Graph().as_default(), tf.Session() as sess:
        X_content = tf.placeholder(tf.float32, shape=output_shape, name="X_content")
        X_pre = vgg.preprocess(X_content)

        if init_img == 'content':
            preds = tf.Variable(np.array([content_target]), dtype=tf.float32)
        elif init_img == 'style':
            preds = tf.Variable(np.array([scale_img(style_target, content_target.shape[0:2])]), dtype=tf.float32)
        else:
            preds = tf.Variable(
             tf.random_normal(output_shape)*75. + 127.
            )
        preds_pre = tf.cast(vgg.preprocess(preds), tf.float32)

        net = vgg.net(vgg_path, preds_pre)

        content_size = tf.cast(tf.reshape(content_features[CONTENT_LAYER], [-1]).shape[0], tf.float32)
        content_loss = content_weight * (2 * tf.nn.l2_loss(
            net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) / content_size
        )

        style_losses = []
        for style_layer in STYLE_LAYERS:
            layer = net[style_layer]
            bs, height, width, filters = map(lambda i:i.value,layer.get_shape())
            size = height * width * filters
            feats = tf.reshape(layer, (bs, height * width, filters))
            feats_T = tf.transpose(feats, perm=[0,2,1])
            grams = tf.matmul(feats_T, feats) / size
            style_gram = style_features[style_layer]
            style_losses.append(2 * tf.nn.l2_loss(grams - style_gram)/style_gram.size)

        style_loss = style_weight * functools.reduce(tf.add, style_losses)

        # total variation denoising
        tv_y_size = tf.cast(_tensor_size(preds[:,1:,:,:]), tf.float32)
        tv_x_size = tf.cast(_tensor_size(preds[:,:,1:,:]), tf.float32)
        y_tv = tf.nn.l2_loss(preds[:,1:,:,:] - preds[:,:output_shape[1]-1,:,:])
        x_tv = tf.nn.l2_loss(preds[:,:,1:,:] - preds[:,:,:output_shape[2]-1,:])
        tv_loss = tv_weight*2*(x_tv/tv_x_size + y_tv/tv_y_size)

        loss = content_loss + style_loss #+ tv_loss

        # overall loss
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        sess.run(tf.global_variables_initializer())
        _preds = sess.run(preds)
        _preds = np.reshape(_preds, content_target.shape)
        save_img(os.path.join(tem_save_path, '1.jpg'), _preds)
        start_time = time.time()
        for step in range(max_steps):
            to_get = [style_loss, content_loss, tv_loss, loss, train_step]
            tup = sess.run(to_get)
            _style_loss,_content_loss,_tv_loss,_loss, ttt = tup
            print "step: {}".format(step)
            print "style_loss: {} content_loss: {} tv_loss: {} loss_ {}".format(_style_loss, \
                    _content_loss, _tv_loss, _loss)
            if (step+1)%save_steps==0:
                _preds = sess.run(preds)
                _preds = np.reshape(_preds, content_target.shape)
                save_img(os.path.join(tem_save_path, '%d.jpg'%(step+1)), _preds)
        end_time = time.time()
        _preds = sess.run(preds)
        _preds = np.reshape(_preds, content_target.shape)
        save_img(output_path, _preds)
        print "optimization done, total time: {}s".format(end_time-start_time)
        print "output image saved at {}".format(output_path)

def _tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d.value for d in tensor.shape[1:]), 1)

def main(args):
    style_target = get_img(args.style_path)
    content_target = get_img(args.content_path)
    os.environ['CUDA_VISIBLE_DIVICES'] = args.gpu_id
    output_path = args.output_path
    tem_save_path = args.tem_save_path
    init_img = args.init_img
    max_steps = args.max_steps
    save_steps = args.save_steps
    content_weight = args.content_weight
    style_weight = args.style_weight
    tv_weight = args.tv_weight
    learning_rate = args.learning_rate
    vgg_path = '../data/imagenet-vgg-verydeep-19.mat'

    optimize(content_target, style_target, content_weight, style_weight, tv_weight,
             learning_rate, vgg_path, max_steps, save_steps, init_img, tem_save_path, output_path)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
    





