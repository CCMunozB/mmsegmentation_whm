import os
import sys

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

from medpy.io import load as loadnii
from medpy.io import save as savenii

name_t1_bet = str(sys.argv[1])
name_t1_bet_mask = str(sys.argv[2])
name_flr = str(sys.argv[3])
name_WMHs = str(sys.argv[4])

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def weight_variable(shape):
    factor = 2
    in_size = shape[2]
    out_size = shape[3]
    stddev = np.sqrt((1.3*factor*2)/(in_size+out_size))
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0., shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, s=1, padding='SAME'):
    return tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding=padding)

def max_pool(x, k=2, s=2, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding=padding)

def dice_coe(output, target, axis=[1,2], smooth=1e-5):
    inse = tf.reduce_sum(tf.multiply(output, target), axis=axis)
    l = tf.reduce_sum(output, axis=axis)
    r = tf.reduce_sum(target, axis=axis)
    dice = (2. * inse + smooth) / (l + r + smooth)
    
    dice = tf.reduce_mean(dice,axis=0)
    
    return dice

def conv_layer(x, in_size, out_size, filter_size=3, is_training=True):
    
    W_conv = weight_variable([filter_size, filter_size, in_size, out_size])
    b_conv = bias_variable([out_size])
    BN_conv = tf.layers.batch_normalization(tf.add(conv2d(x, W_conv), b_conv), training=is_training)
    h_conv = tf.nn.elu(BN_conv)
    
    return h_conv

def final_layer(x, in_size, out_size, filter_size=1):
    
    W_conv = weight_variable([filter_size, filter_size, in_size, out_size])
    b_conv = bias_variable([out_size])
    h_score = tf.add(conv2d(x, W_conv), b_conv)
    output = tf.nn.softmax(h_score)
    
    return output
    
def z_score(data, lth = 0.02, uth = 0.98):
    
    temp = np.sort(data[data>0])
    lth_num = int(temp.shape[0]*0.02)
    uth_num = int(temp.shape[0]*0.98)
    data_mean = np.mean(temp[lth_num:uth_num])
    data_std = np.std(temp[lth_num:uth_num])
    data = (data - data_mean)/data_std
    
    return data

def upsampling2d(x, out):
    
    x_shape = tf.shape(x)
    out_shape = tf.shape(out)
        
    up_image = tf.image.resize_nearest_neighbor(x, (out_shape[1], out_shape[2]))
    
    return up_image

# data read
T1_bet_data, T1_bet_header = loadnii(name_t1_bet)
T1_bet_mask_data, T1_bet_mask_header = loadnii(name_t1_bet_mask)

FLAIR_data, FLAIR_header = loadnii(name_flr)
FLAIR_bet_data = FLAIR_data*T1_bet_mask_data

# channel size
channel = 2 # T1 & FLAIR

# the number of label
label_num = 2 # background or foreground(WMH)

### data preprocessing

ml = 500 # max length
cs = 100 # crop size; cs*2 = crop size

#
axial_sum_nonzero = np.where(np.sum(FLAIR_bet_data, axis=(0, 1)) > 0)

z_axis_min = np.min(axial_sum_nonzero)
z_axis_max = np.max(axial_sum_nonzero)

#
T1_data_zcrop = T1_bet_data[:,:,z_axis_min:z_axis_max+1]
FLAIR_data_zcrop = FLAIR_bet_data[:,:,z_axis_min:z_axis_max+1]

#
T1_data_zscore = z_score(T1_data_zcrop)
FLAIR_data_zscore = z_score(FLAIR_data_zcrop)

#
x_s, y_s, z_s = FLAIR_data_zscore.shape[0], FLAIR_data_zscore.shape[1], FLAIR_data_zscore.shape[2]

min_x_axis = int(ml/2-x_s/2)
max_x_axis = int(min_x_axis + x_s)
min_y_axis = int(ml/2-y_s/2)
max_y_axis = int(min_y_axis + y_s)

T1_data_xyCtr = np.zeros((ml,ml,z_s))
T1_data_xyCtr[min_x_axis:max_x_axis, min_y_axis:max_y_axis, :] = T1_data_zscore
T1_data_xyCtr = T1_data_xyCtr[int(ml/2-cs):int(ml/2+cs), int(ml/2-cs):int(ml/2+cs), :]

FLAIR_data_xyCtr = np.zeros((ml,ml,z_s))
FLAIR_data_xyCtr[min_x_axis:max_x_axis, min_y_axis:max_y_axis, :] = FLAIR_data_zscore
FLAIR_data_xyCtr = FLAIR_data_xyCtr[int(ml/2-cs):int(ml/2+cs), int(ml/2-cs):int(ml/2+cs), :]

#
T1_data_swp = np.swapaxes(T1_data_xyCtr, 0, 2)
FLAIR_data_swp = np.swapaxes(FLAIR_data_xyCtr, 0, 2)

#
T1_data_rsp = np.reshape(T1_data_swp,(z_s, cs*2, cs*2, 1))
FLAIR_data_rsp = np.reshape(FLAIR_data_swp,(z_s, cs*2, cs*2, 1))
            
concat_data = np.concatenate((T1_data_rsp, FLAIR_data_rsp), axis=3)

### tensorflow, 2D UNet

tf.reset_default_graph()

# placeholder definination of image and label information 
X = tf.placeholder(tf.float32, shape=[None, None, None, channel])
Y = tf.placeholder(tf.float32, shape=[None, None, None, label_num])
is_training = tf.placeholder(tf.bool, name='MODE')
lr = tf.placeholder("float")

Y_sub_2 = max_pool(Y)
Y_fore_2, Y_back_2 = tf.split(Y_sub_2, [1, 1], 3)
Y_back_2 = 1 - Y_fore_2
Y_2 = tf.concat([Y_fore_2, Y_back_2], 3)

Y_fore_3 = max_pool(Y_fore_2)
Y_back_3 = 1 - Y_fore_3
Y_3 = tf.concat([Y_fore_3, Y_back_3], 3)

Y_fore_4 = max_pool(Y_fore_3)
Y_back_4 = 1 - Y_fore_4
Y_4 = tf.concat([Y_fore_4, Y_back_4], 3)

Y_fore_5 = max_pool(Y_fore_4)
Y_back_5 = 1 - Y_fore_5
Y_5 = tf.concat([Y_fore_5, Y_back_5], 3)

# Encoder
conv_1 = conv_layer(X, channel, 64, is_training=is_training, filter_size=5)
conv_2 = conv_layer(conv_1, 64, 64, is_training=is_training, filter_size=5)
hpool_1 = max_pool(conv_2)

conv_3 = conv_layer(hpool_1, 64, 96, is_training=is_training)
conv_4 = conv_layer(conv_3, 96, 96, is_training=is_training)
hpool_2 = max_pool(conv_4)

conv_5 = conv_layer(hpool_2, 96, 128, is_training=is_training)
conv_6 = conv_layer(conv_5, 128, 128, is_training=is_training)
hpool_3 = max_pool(conv_6)

conv_7 = conv_layer(hpool_3, 128, 256, is_training=is_training)
conv_8 = conv_layer(conv_7, 256, 256, is_training=is_training)
hpool_4 = max_pool(conv_8)

conv_9 = conv_layer(hpool_4, 256, 512, is_training=is_training)
conv_10 = conv_layer(conv_9, 512, 512, is_training=is_training)

# Decoder

dconv1 = upsampling2d(conv_10, conv_8)
conv_concat = tf.concat([dconv1, conv_8], 3)

conv_11 = conv_layer(conv_concat, 768, 256, is_training=is_training)
conv_12 = conv_layer(conv_11, 256, 256, is_training=is_training)

dconv2 = upsampling2d(conv_12, conv_6)
conv_concat = tf.concat([dconv2, conv_6], 3)

conv_13 = conv_layer(conv_concat, 384, 128, is_training=is_training)
conv_14 = conv_layer(conv_13, 128, 128, is_training=is_training)

dconv3 = upsampling2d(conv_14, conv_4)
conv_concat = tf.concat([dconv3, conv_4], 3)

conv_15 = conv_layer(conv_concat, 224, 96, is_training=is_training)
conv_16 = conv_layer(conv_15, 96, 96, is_training=is_training)

dconv4 = upsampling2d(conv_16, conv_2)
conv_concat = tf.concat([dconv4, conv_2], 3)

conv_17 = conv_layer(conv_concat, 160, 64, is_training=is_training)
conv_18 = conv_layer(conv_17, 64, 64, is_training=is_training)

# Output
output_1 = final_layer(conv_18, 64, 2)
output_2 = final_layer(conv_16, 96, 2)
output_3 = final_layer(conv_14, 128, 2)
output_4 = final_layer(conv_12, 256, 2)
output_5 = final_layer(conv_10, 512, 2)

# Segmentation results & Dice
output2 = tf.cast(output_1 > 0.5, dtype=tf.float32)

# Dice loss function
loss_1 = 2 - dice_coe(output_1, Y)[0] - dice_coe(output_1,Y)[1] # 0 -> fg , 1 -> bg
loss_2 = 2 - dice_coe(output_2, Y_2)[0] - dice_coe(output_2,Y_2)[1] # 0 -> fg , 1 -> bg
loss_3 = 2 - dice_coe(output_3, Y_3)[0] - dice_coe(output_3,Y_3)[1] # 0 -> fg , 1 -> bg
loss_4 = 2 - dice_coe(output_4, Y_4)[0] - dice_coe(output_4,Y_4)[1] # 0 -> fg , 1 -> bg
loss_5 = 2 - dice_coe(output_5, Y_5)[0] - dice_coe(output_5,Y_5)[1] # 0 -> fg , 1 -> bg

# Adam optimizer
update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(lr).minimize(0.2*loss_1+0.2*loss_2+0.2*loss_3+0.2*loss_4+0.2*loss_5)

# Session run
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

### WMHs segmentation

seg_ensemble = 0

for fold_num in range(1, 6):
    
        saver.restore(sess, "tools_wmh/WMHseg_MICCAI2_2D_UNet_default_0712_%d.ckpt"%(fold_num))
                
        concat_data_x = np.flip(concat_data, 1)
        concat_data_y = np.flip(concat_data, 2)
        concat_data_xy = np.flip(np.flip(concat_data, 1), 2)

        seg_results = sess.run(output2, feed_dict={X:concat_data, is_training: False})
        seg_results = seg_results[:,:,:,0]

        seg_results_x = sess.run(output2, feed_dict={X:concat_data_x, is_training: False})
        seg_results_x = np.flip(seg_results_x[:,:,:,0], 1)

        seg_results_y = sess.run(output2, feed_dict={X:concat_data_y, is_training: False})
        seg_results_y = np.flip(seg_results_y[:,:,:,0], 2)

        seg_results_xy = sess.run(output2, feed_dict={X:concat_data_xy, is_training: False})
        seg_results_xy = np.flip(np.flip(seg_results_xy[:,:,:,0], 1), 2)

        seg_results_FE = (seg_results + seg_results_x + seg_results_y + seg_results_xy)/4.0
        seg_ensemble += seg_results_FE


seg_ensemble = seg_ensemble/5.0
seg_ensemble = (seg_ensemble >= 0.5)

### WMHs save
x_s, y_s, z_s = FLAIR_data.shape[0], FLAIR_data.shape[1], FLAIR_data.shape[2]

seg_ensemble = np.swapaxes(seg_ensemble, 0, 2)

seg_ensemble_WMHs = np.zeros((ml,ml,z_s))
seg_ensemble_WMHs[int(ml/2-cs):int(ml/2+cs), int(ml/2-cs):int(ml/2+cs), z_axis_min:z_axis_max+1] = seg_ensemble
seg_ensemble_WMHs = seg_ensemble_WMHs[min_x_axis:max_x_axis, min_y_axis:max_y_axis, :]

savenii(seg_ensemble_WMHs, name_WMHs, FLAIR_header)