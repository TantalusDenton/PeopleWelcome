
import numpy as np
import PIL
import tensorflow as tf
import tensorflow_hub as hub
import os

from tensorflow.python.framework import ops
import tensorflow as tf
slim = tf.contrib.slim

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from resnet_v2 import resnet_v2_50


class MultiLabelClassificationModel(object):
	"""multi label image classification """


	def __init__(self, model_config):
		"""multi lable classification initializer.
		Args:
			model_config: instance if ModelConfig class.
		"""
		# config
		self.model_config = model_config

		# [batch_size, height, width, channels]
		self.images = None

		# str of image content
		self.image_feed = None

		# [batch_size, label_count]
		self.target_labels = None

		# restore the base network from checkpoint
		self.base_network_init_fn = None

		# Global Tensor
		self.global_step = None

		# used in inference
		self.sigmoid_result = None

		# loss 
		self.sigmoid_cross_entropy_loss = None 
		self.total_loss = None


	def process_image(self, encoded_image, thread_id=0):
		"""Decodes and processes an image string.
		"""
		with tf.name_scope("decode_image", values=[encoded_image]):
			image = tf.image.decode_jpeg(encoded_image, channels=3)
			image = tf.image.convert_image_dtype(image, dtype=tf.float32)

			if self.model_config.mode == "train":
				# resize 
				image = tf.image.resize_images(image, 
					size=[self.model_config.resize_height, self.model_config.resize_width], 
					method=tf.image.ResizeMethod.BILINEAR)
			
				# crop to size
				image = tf.random_crop(image, 
					[self.model_config.target_height, self.model_config.target_width, 3])
				
				image = tf.image.random_flip_left_right(image)
				image = tf.image.random_brightness(image, max_delta=32. / 255.)
				image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
				image = tf.image.random_hue(image, max_delta=0.032)
				image = tf.image.random_contrast(image, lower=0.5, upper=1.5)

				image = tf.clip_by_value(image, 0.0, 1.0)

			else:
				image = tf.image.resize_images(image, 
					size=[self.model_config.target_height, self.model_config.target_width], 
					method=tf.image.ResizeMethod.BILINEAR)

			# Rescale to [-1, 1] instead of [0, 1]
			image = tf.subtract(image, 0.5)
			image = tf.multiply(image, 2.0)

			return image

	def over_sample_image(self, encoded_image):
		image = tf.image.decode_jpeg(encoded_image, channels=3)
		image = tf.image.convert_image_dtype(image, dtype=tf.float32)
		image = tf.image.resize_images(image, 
			size=[self.model_config.resize_height, self.model_config.resize_width], 
			method=tf.image.ResizeMethod.BILINEAR)
	
		im_shape = tf.shape(image)
		crop1 = tf.slice(image, 
			[0, 0, 0], 
			[self.model_config.target_height, self.model_config.target_width, 3], 
			name="crop_1")

		crop2 = tf.slice(image, 
			[0, self.model_config.resize_width - self.model_config.target_width, 0], 
			[self.model_config.target_height, self.model_config.target_width, 3], 
			name="crop_2")

		crop3 = tf.slice(image, 
			[self.model_config.resize_height - self.model_config.target_height, 0, 0], 
			[self.model_config.target_height, self.model_config.target_width, 3], 
			name="crop_3")

		crop4 = tf.slice(image, 
			[self.model_config.resize_height - self.model_config.target_height, self.model_config.resize_width - self.model_config.target_width, 0], 
			[self.model_config.target_height, self.model_config.target_width, 3], 
			name="crop_4")

		crop5 = tf.slice(image, 
			[(self.model_config.resize_height - self.model_config.target_height)/2, (self.model_config.resize_width - self.model_config.target_width)/2, 0], 
			[self.model_config.target_height, self.model_config.target_width, 3], 
			name="crop_5")

		reverse1 = tf.reverse(crop1, [1], name="reverse_1")
		reverse2 = tf.reverse(crop2, [1], name="reverse_2")
		reverse3 = tf.reverse(crop3, [1], name="reverse_3")
		reverse4 = tf.reverse(crop4, [1], name="reverse_4")
		reverse5 = tf.reverse(crop5, [1], name="reverse_5")

		images = tf.stack([
			crop1,
			crop2,
			crop3,
			crop4,
			crop5,
			reverse1,
			reverse2,
			reverse3,
			reverse4,
			reverse5])

		return images
		

	def build_inputs(self):
		"""Read, preprocessing, and batching
		Outputs:
			self.image_feed
			self.images
			self.target_labels
		"""
		if self.model_config.mode == "inference":
			# image as string
			self.image_feed = tf.placeholder(dtype=tf.string, shape=[], name="image_feed")

			#image = self.process_image(self.image_feed)

			#images = tf.expand_dims(image, 0)

			images = self.over_sample_image(self.image_feed)

			#tf.logging.info("images type: {}, shape: {}".format(type(images), images.shape))
			
            target_labels = None
		else:
			filename_queue = tf.train.string_input_producer([self.model_config.input_tfrecord_file])
			reader = tf.TFRecordReader()
			_, serialized_example = reader.read(filename_queue)
			
			#sequence example
			context, sequence = tf.parse_single_sequence_example(
				serialized_example,
				context_features={
					self.model_config.image_path_key: tf.FixedLenFeature([], dtype=tf.string),
					self.model_config.image_data_key: tf.FixedLenFeature([], dtype=tf.string),
				},
				sequence_features={
					self.model_config.image_label_key: tf.FixedLenSequenceFeature([], dtype=tf.int64),
				})		

			image_path = context[self.model_config.image_path_key]
			encoded_image = context[self.model_config.image_data_key]
			image_label = sequence[self.model_config.image_label_key]
			image_label = tf.cast(image_label, tf.float32)
			
			image = self.process_image(encoded_image)
			
			if self.model_config.mode == "train":
				min_after_dequeue = self.model_config.batch_size * 3
				capacity = min_after_dequeue + (1 + self.model_config.num_threads) * self.model_config.batch_size
				images, target_labels = tf.train.shuffle_batch(
					[image, image_label],
					self.model_config.batch_size,
					min_after_dequeue=min_after_dequeue,
					num_threads=self.model_config.num_threads,
					capacity=capacity,
					shapes=[[self.model_config.target_height, self.model_config.target_width, 3], [self.model_config.label_count]])
					
			else:
				capacity = (1 + self.model_config.num_threads) * self.model_config.batch_size
				images, target_labels = tf.train.batch(
					[image, image_label],
					self.model_config.batch_size,
					num_threads=self.model_config.num_threads,
					capacity=capacity,
					dynamic_pad=True)

		self.images = images
		self.target_labels = target_labels


	def resnet_arg_scope(self):
		""""Defines the default ResNet arg scope
		
		Args:
			self.model_config.train_base_network, self.model_config.mode
		Return:
			resnet arg scope
		"""
        weight_decay=0.0001 
		activation_fn=tf.nn.relu
		use_batch_norm=True
		is_base_network_training = self.model_config.train_base_network and (self.model_config.mode == "train")
		batch_norm_decay=0.997
		batch_norm_epsilon=1e-5
		batch_norm_scale=True
		batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS
		
		if use_batch_norm:
			batch_norm_params = {
                'decay': batch_norm_decay,
				'epsilon': batch_norm_epsilon,
				'scale': batch_norm_scale,
				"is_training": is_base_network_training,
				"trainable": self.model_config.train_base_network,
				'updates_collections': batch_norm_updates_collections,
				'fused': None,  # Use fused batch norm if possible.
			}
		else:
			batch_norm_params = {}

		if self.model_config.train_base_network:
			weights_regularizer = slim.l2_regularizer(weight_decay)
		else:
			weights_regularizer = None

		with slim.arg_scope([slim.conv2d],
			trainable=self.model_config.train_base_network,
			weights_regularizer=weights_regularizer,
			weights_initializer=slim.variance_scaling_initializer(),
			activation_fn=activation_fn,
			normalizer_fn=slim.batch_norm if use_batch_norm else None,
			normalizer_params=batch_norm_params):

			with slim.arg_scope([slim.batch_norm], **batch_norm_params):
				with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
					return arg_sc

					
	def build_model(self):
		"""Builds model.
		Inputs:
			self.images
			self.target_labels
			self.model_config.train_base_network
			self.model_config.mode
		Outputs:
			self.sigmoid_result (inference)
			self.cross_entropy (train, eval)
			self.total_loss
		"""

		arg_scope = self.resnet_arg_scope()
		is_base_network_training = self.model_config.train_base_network and (self.model_config.mode == "train")
		with slim.arg_scope(arg_scope):
			worknet, end_points = resnet_v2_50(self.images, is_training=is_base_network_training, global_pool=False)

		with tf.variable_scope('my_sub_network'):
			if self.model_config.use_regularizer:
				weights_regularizer = slim.l2_regularizer(self.model_config.weight_decay)
			else:
				weights_regularizer = None

			worknet = slim.conv2d(worknet, 
				self.model_config.label_count, 
				[1, 1], 
				weights_regularizer=weights_regularizer, 
				scope="my_conv_1")

			worknet = slim.conv2d(worknet, 
				self.model_config.label_count, 
				[1, 1], 
				weights_regularizer=weights_regularizer, 
				scope="my_conv_2")

			worknet = slim.conv2d(worknet, 
				self.model_config.label_count, [7, 7], 
				weights_regularizer=weights_regularizer, 
				padding="VALID", 
				activation_fn=None, 
				scope="my_conv_3")

			logits = tf.squeeze(worknet, [1, 2], name="logits")
		
		self.sigmoid_result = tf.sigmoid(logits, name="sigmoid_result")

		if self.model_config.mode == "inference":
			return
		
		tf.logging.info("Before:--- GraphKeys.LOSSES len: {}; GraphKeys.REGULARIZATION_LOSSES len: {}".format(len(tf.losses.get_losses()), len(tf.losses.get_regularization_losses())))

		# By default, the losses in 'tf.losses' are collected into the GraphKeys.LOSSES collection.
		sigmoid_cross_entropy_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.target_labels, logits=logits, scope="sigmoid_cross_entropy")

		total_loss = tf.losses.get_total_loss()

		tf.logging.info("After:--- GraphKeys.LOSSES len: {}; GraphKeys.REGULARIZATION_LOSSES len: {}".format(len(tf.losses.get_losses()), len(tf.losses.get_regularization_losses())))

		# Add summaries.
		tf.summary.scalar("losses/sigmoid_cross_entropy_loss", sigmoid_cross_entropy_loss)
		tf.summary.scalar("losses/total_loss", total_loss)

		self.sigmoid_cross_entropy_loss = sigmoid_cross_entropy_loss 
		self.total_loss = total_loss

		
	def base_network_initializer(self):
		"""Sets up the function to restore base network variables from checkpoint."""
		if self.model_config.mode != "inference" and self.model_config.base_network_checkpoint != None:
			# restore base network model
			tf.logging.info("Restoring resnet_v2_50 from checkpoint file: {}".format(self.model_config.base_network_checkpoint))
			base_network_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="resnet_v2_50")
			saver = tf.train.Saver(base_network_variables)
			def restore_fn(sess):
				saver.restore(sess, self.model_config.base_network_checkpoint)

			self.base_network_init_fn = restore_fn


	def setup_global_step(self):
		"""Sets up the global step Tensorflow."""
		self.global_step = tf.Variable(
			initial_value=0,
			name="global_step",
			trainable=False,
			collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])


	def build(self):
		"""Creates all train, eval and inference."""
		self.build_inputs()
		self.build_model()
		
		self.base_network_initializer()
		self.setup_global_step()


class ModelConfig(object):
	def __init__(self, mode, label_count, base_network_checkpoint=None, input_tfrecord_file=None):
		"""Initialize model config.
			check mode, label count, base network checkpoint,
            input record files
		"""
		self.mode = mode

		self.label_count = label_count

		self.base_network_checkpoint = base_network_checkpoint

		self.input_tfrecord_file = input_tfrecord_file

		# target size
		self.target_height = 224
		self.target_width = 224

		self.resize_height = 255
		self.resize_width = 255

		self.use_regularizer= True
		self.weight_decay = 0.0001
		
		# batch 
		self.batch_size = 32

		# base network
		self.train_base_network = False
		
		# sequence example 
		self.image_path_key = "image_path"
		self.image_data_key = "image_data"
		self.image_label_key = "image_label"

		self.num_threads = 1
