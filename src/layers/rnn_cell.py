# _*_coding:utf8_*_
# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Module for constructing RNN Cells."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('/home/liujie3/e3d_lstm-master')

import tensorflow as tf
import tensorflow.contrib.layers as layers   # tensorflow1.X版本，2以上已经舍弃？？


class EideticLSTMCell(object):
  """Eidetic LSTM recurrent network cell.

  Implements the model as described in
  Wang, Yunbo, et al. "Eidetic 3D LSTM: A Model for Video Prediction and
  Beyond.", ICLR (2019). https://openreview.net/pdf?id=B1lKS2AqtX
  """

  def __init__(self,                  
               conv_ndims,          
               input_shape,
               output_channels,   
               kernel_shape,
               layer_norm=True,
               norm_gain=1.0,
               norm_shift=0.0,
               forget_bias=1.0,
               name="eidetic_lstm_cell"):
    """Construct EideticLSTMCell.

    Args:
      conv_ndims: Convolution dimensionality (1, 2 or 3).
      input_shape: Shape of the input as int tuple, excluding the batch size.
      output_channels: int, number of output channels of the conv LSTM.
      kernel_shape: Shape of kernel as in tuple (of size 1,2 or 3).
      layer_norm: If `True`, layer normalization will be applied.
      norm_gain: float, The layer normalization gain initial value. If
        `layer_norm` has been set to `False`, this argument will be ignored.
      norm_shift: float, The layer normalization shift initial value. If
        `layer_norm` has been set to `False`, this argument will be ignored.
      forget_bias: Forget bias.
      name: Name of the module.

    Raises:
      ValueError: If `input_shape` is incompatible with `conv_ndims`.
    """
    # super(EideticLSTMCell, self).__init__(name=name)

    if conv_ndims != len(input_shape) - 1:         
      raise ValueError("Invalid input_shape {} for conv_ndims={}.".format(
          input_shape, conv_ndims))

    self._conv_ndims = conv_ndims
    self._input_shape = input_shape    # (29,2,29,256)
    self._output_channels = output_channels    # 64
    self._kernel_shape = kernel_shape    # (2,5,5)
    self._layer_norm = layer_norm
    self._norm_gain = norm_gain   # 1.0  
    self._norm_shift = norm_shift  # 0.0   
    self._forget_bias = forget_bias   # 1.0
    self._layer_name = name

    self._state_size = tf.TensorShape(self._input_shape[:-1] +   
                                      [self._output_channels])
    self._output_size = tf.TensorShape(self._input_shape[:-1] +
                                       [self._output_channels])
    self._initializer = tf.contrib.layers.xavier_initializer()

  @property
  def output_size(self):
    return self._output_size

  @property
  def state_size(self):
    return self._state_size

  def _norm(self, inp, scope, dtype=tf.float32):     
    shape = inp.get_shape()[-1:] 
    gamma_init = tf.constant_initializer(self._norm_gain)  
    beta_init = tf.constant_initializer(self._norm_shift)
    with tf.variable_scope(scope):         
      # Initialize beta and gamma for use by layer_norm.     # get_variable(name,shape,initializer,dtype)
      tf.get_variable("gamma", shape=shape, initializer=gamma_init, dtype=dtype)
      tf.get_variable("beta", shape=shape, initializer=beta_init, dtype=dtype)
    normalized = layers.layer_norm(inp, reuse=True, scope=scope)  
    return normalized                                              # import tensorflow.contrib.layers as layers

  def _attn(self, in_query, in_keys, in_values):     
    """3D Self-Attention Block.                     
    Args:                                            
      in_query: Tensor of shape (b,l,w,h,n).          
      in_keys: Tensor of shape (b,attn_length,w,h,n).     
      in_values: Tensor of shape (b,attn_length,w,h,n).

    Returns:
      attn: Tensor of shape (b,l,w,h,n).

    Raises:
      ValueError: If any number of dimensions regarding the inputs is not 4 or 5
        or if the corresponding dimension lengths of the inputs are not
        compatible.
    """
    q_shape = in_query.get_shape().as_list()   
    if len(q_shape) == 4:     
      batch = q_shape[0]
      width = q_shape[1]
      height = q_shape[2]
      num_channels = q_shape[3]
    elif len(q_shape) == 5:
      batch = q_shape[0]
      width = q_shape[2]
      height = q_shape[3]
      num_channels = q_shape[4]
    else:
      raise ValueError("Invalid input_shape {} for the query".format(q_shape))

    k_shape = in_keys.get_shape().as_list()   
    if len(k_shape) != 5:
      raise ValueError("Invalid input_shape {} for the keys".format(k_shape))

    v_shape = in_values.get_shape().as_list()    
    if len(v_shape) != 5:
      raise ValueError("Invalid input_shape {} for the values".format(v_shape))
    
    if width != k_shape[2] or height != k_shape[3] or num_channels != k_shape[4]:
      raise ValueError("Invalid input_shape {} and {}, not compatible.".format(
          q_shape, k_shape))
    if width != v_shape[2] or height != v_shape[3] or num_channels != v_shape[4]:
      raise ValueError("Invalid input_shape {} and {}, not compatible.".format(
          q_shape, v_shape))
    if k_shape[2] != v_shape[2] or k_shape[3] != v_shape[3] or k_shape[
        4] != v_shape[4]:
      raise ValueError("Invalid input_shape {} and {}, not compatible.".format(
          k_shape, v_shape))
    # batch、num_channels  来自 in_query
    query = tf.reshape(in_query, [batch, -1, num_channels])    #
    keys = tf.reshape(in_keys, [batch, -1, num_channels])     
    values = tf.reshape(in_values, [batch, -1, num_channels])  
    attn = tf.matmul(query, keys, False, True) 
    attn = tf.nn.softmax(attn, axis=2)
    attn = tf.matmul(attn, values, False, False) 
    if len(q_shape) == 4:
      attn = tf.reshape(attn, [batch, width, height, num_channels])
    else:
      attn = tf.reshape(attn, [batch, -1, width, height, num_channels]) 
    return attn

  def _conv(self, inputs, output_channels, kernel_shape):
    if self._conv_ndims == 2:
      return tf.layers.conv2d( 
          inputs, output_channels, kernel_shape, padding="same")
    elif self._conv_ndims == 3:
      return tf.layers.conv3d(
          inputs, output_channels, kernel_shape, padding="same")

  def _myatten(self, H):
    H_shape = H.get_shape().as_list()
    if len(H_shape) == 4:  
      batch = H_shape[0]
      width = H_shape[1]
      height = H_shape[2]
      num_channels = H_shape[3]
    elif len(H_shape) == 5:
      batch = H_shape[0]
      width = H_shape[2]
      height = H_shape[3]
      num_channels = H_shape[4]
    else:
      raise ValueError("Invalid input_shape {} for the query".format(H_shape))
    sa_size = 256
    #
    H_reshape= tf.reshape(H, [batch, -1, num_channels])
    W_s1 = tf.get_variable("W_s1", shape=[batch, num_channels, H_reshape.shape[1]], initializer=self._initializer) # (1,64,256)
    _H_s1 = tf.nn.tanh(tf.matmul(W_s1, H_reshape))
    W_s2 = tf.get_variable("W_s2", shape=[batch, num_channels, num_channels], initializer=self._initializer)  # (1,256,64)
    _H_s2 = tf.matmul(_H_s1, W_s2)
    A = tf.nn.softmax(_H_s2, name="attention")
    H_hat = tf.matmul(H_reshape,A)
    H_hat = tf.reshape(H_hat, [batch, -1, width, height, num_channels])
    return H_hat



  def __call__(self, inputs, hidden, cell, global_memory, eidetic_cell):  
    with tf.variable_scope(self._layer_name):
      new_hidden = self._conv(hidden, 4 * self._output_channels,
                              self._kernel_shape)
      if self._layer_norm:
        new_hidden = self._norm(new_hidden, "hidden") 
      i_h, g_h, r_h, o_h = tf.split(
          value=new_hidden, num_or_size_splits=4, axis=-1) 

      new_inputs = self._conv(inputs, 7 * self._output_channels, 
                              self._kernel_shape)
      if self._layer_norm:
        new_inputs = self._norm(new_inputs, "inputs") 
        i_x, g_x, r_x, o_x, temp_i_x, temp_g_x, temp_f_x = tf.split(
            value=new_inputs, num_or_size_splits=7, axis=-1)   

      i_t = tf.sigmoid(i_x + i_h)  
      r_t = tf.sigmoid(r_x + r_h)
      g_t = tf.tanh(g_x + g_h)  

      new_cell = r_t * cell + i_t * g_t



      new_global_memory = self._conv(global_memory, 4 * self._output_channels,  
                                     self._kernel_shape)


      if self._layer_norm:
        new_global_memory = self._norm(new_global_memory, "global_memory")
        i_m, f_m, g_m, m_m = tf.split(                  
            value=new_global_memory, num_or_size_splits=4, axis=-1)

      temp_i_t = tf.sigmoid(temp_i_x + i_m)             
      temp_f_t = tf.sigmoid(temp_f_x + f_m + self._forget_bias)
      temp_g_t = tf.tanh(temp_g_x + g_m)
      new_global_memory = temp_f_t * tf.tanh(m_m) + temp_i_t * temp_g_t 

      o_c = self._conv(new_cell, self._output_channels, self._kernel_shape)
      o_m = self._conv(new_global_memory, self._output_channels,
                       self._kernel_shape)

      output_gate = tf.tanh(o_x + o_h + o_c + o_m) 

      memory = tf.concat([new_cell, new_global_memory], -1)    
      memory = self._conv(memory, self._output_channels, 1)    

      output = tf.tanh(memory) * tf.sigmoid(output_gate)   
      #
      #output = self._myatten(output)  # (1,2,29,29,64)

    return output, new_cell, new_global_memory  


class Eidetic2DLSTMCell(EideticLSTMCell):
  """2D Eidetic LSTM recurrent network cell.

  Implements the model as described in
  Wang, Yunbo, et al. "Eidetic 3D LSTM: A Model for Video Prediction and
  Beyond.", ICLR (2019). https://openreview.net/pdf?id=B1lKS2AqtX
  """

  def __init__(self, name="eidetic_2d_lstm_cell", **kwargs):
    """Construct Eidetic2DLSTMCell. See `EideticLSTMCell` for more details."""
    super(Eidetic2DLSTMCell, self).__init__(conv_ndims=2, name=name, **kwargs)


class Eidetic3DLSTMCell(EideticLSTMCell):
  """3D Eidetic LSTM recurrent network cell.

  Implements the model as described in
  Wang, Yunbo, et al. "Eidetic 3D LSTM: A Model for Video Prediction and
  Beyond.", ICLR (2019). https://openreview.net/pdf?id=B1lKS2AqtX
  """

  def __init__(self, name="eidetic_3d_lstm_cell", **kwargs):  
    """Construct Eidetic3DLSTMCell. See `EideticLSTMCell` for more details."""
    super(Eidetic3DLSTMCell, self).__init__(conv_ndims=3, name=name, **kwargs)  
