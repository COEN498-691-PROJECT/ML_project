
from typing import Dict, List, Text

import os
import glob
from absl import logging

import datetime
import tensorflow as tf
import tensorflow_transform as tft

from tfx import v1 as tfx
from tfx_bsl.public import tfxio
from tensorflow_transform import TFTransformOutput
# constants are cached so for updated, need to reload imports and file
import constants
import sys
if 'google.colab' in sys.modules:  # Testing to see if we're doing development
  import importlib
  importlib.reload(constants)


_LABEL_KEY = constants.LABEL_KEY

_BATCH_SIZE = 40

# features and label for tuning/training
def input_fn(file_pattern: List[Text],
              data_accessor: tfx.components.DataAccessor,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 200) -> tf.data.Dataset:
  
  return data_accessor.tf_dataset_factory(
      file_pattern,
      tfxio.TensorFlowDatasetOptions(
          batch_size=batch_size, label_key=_LABEL_KEY),
      tf_transform_output.transformed_metadata.schema)

# serving default signature
def get_tf_examples_serving_signature(model, tf_transform_output):

  # to track the layers in the model in order to save it
  model.tft_layer_inference = tf_transform_output.transform_features_layer()

  @tf.function(input_signature=[
      tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
  ])
  def serve_tf_examples_fn(serialized_tf_example):
    raw_feature_spec = tf_transform_output.raw_feature_spec()
    # Remove label feature since these will not be present at serving time
    raw_feature_spec.pop(_LABEL_KEY)
    raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
    transformed_features = model.tft_layer_inference(raw_features)
    logging.info('serve_transformed_features = %s', transformed_features)

    outputs = model(transformed_features)
    # returns outputs to use in serving sign
    return {'outputs': outputs}

  return serve_tf_examples_fn

# transform features, used by tfx validator
def get_transform_features_signature(model, tf_transform_output):

  # To track the layers in the model in order to save it
  model.tft_layer_eval = tf_transform_output.transform_features_layer()

  @tf.function(input_signature=[
      tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
  ])
  def transform_features_func(serialized_tf_example):
    raw_feature_spec = tf_transform_output.raw_feature_spec()
    raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
    transformed_features = model.tft_layer_eval(raw_features)
    logging.info('eval_transformed_features = %s', transformed_features)
    # returns the transformed_features to be fed as input to evaluator
    return transformed_features

  return transform_features_func


# export keras model with 2 signatures, default and transform feature
def export_serving_model(tf_transform_output, model, output_dir):
  # layer saved to the model for keras tracking purpases
  model.tft_layer = tf_transform_output.transform_features_layer()

  signatures = {
      'serving_default':
          get_tf_examples_serving_signature(model, tf_transform_output),
      'transform_features':
          get_transform_features_signature(model, tf_transform_output),
  }

  tf.saved_model.save(model, output_dir, signatures=signatures)


# build DNN keras model for classifying 4 HAR activities
def build_keras_model(tf_transform_output: TFTransformOutput) -> tf.keras.Model:
  feature_spec = tf_transform_output.transformed_feature_spec().copy()
  feature_spec.pop(_LABEL_KEY)

  inputs = {}
  for key, spec in feature_spec.items():
      if isinstance(spec, tf.io.VarLenFeature):
          inputs[key] = tf.keras.layers.Input(
              shape=[None], name=key, dtype=spec.dtype, sparse=True)
      elif isinstance(spec, tf.io.FixedLenFeature):
          inputs[key] = tf.keras.layers.Input(
              shape=spec.shape or [1], name=key, dtype=spec.dtype)
      else:
          raise ValueError('Spec type is not supported: ', key, spec)

  x = tf.keras.layers.Concatenate()(tf.nest.flatten(inputs))
  x = tf.keras.layers.Dense(100, activation='relu')(x)
  x = tf.keras.layers.Dense(70, activation='relu')(x)
  x = tf.keras.layers.Dense(50, activation='relu')(x)
  x = tf.keras.layers.Dense(20, activation='relu')(x)
  output = tf.keras.layers.Dense(4)(x)  # 4 logits for 4 activities
  return tf.keras.Model(inputs=inputs, outputs=output)


# TFX Trainer
def run_fn(fn_args: tfx.components.FnArgs):
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  train_dataset = input_fn(fn_args.train_files, fn_args.data_accessor,
                            tf_transform_output, _BATCH_SIZE)
  eval_dataset = input_fn(fn_args.eval_files, fn_args.data_accessor,
                           tf_transform_output, _BATCH_SIZE)

  model = build_keras_model(tf_transform_output)
    
  # compile model with multiclass metrics from tf keras metrics
  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
      metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(name='overall_accuracy'),
        # per-class accuracy
        tf.keras.metrics.SparseCategoricalAccuracy(name='lying_accuracy'),
        tf.keras.metrics.SparseCategoricalAccuracy(name='walking_accuracy'),
        tf.keras.metrics.SparseCategoricalAccuracy(name='sitting_accuracy'),
        tf.keras.metrics.SparseCategoricalAccuracy(name='running_accuracy'),])

  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=fn_args.model_run_dir, update_freq='batch')

  model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps,
      callbacks=[tensorboard_callback])

  # Export the model.
  export_serving_model(tf_transform_output, model, fn_args.serving_model_dir)
