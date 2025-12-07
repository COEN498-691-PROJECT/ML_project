
import tensorflow as tf
import tensorflow_transform as tft

# Imported files such as constants are normally cached, so changes are
# not honored after the first import
# Can be prob in development when we may be iterating code
# To avoid this problem during development, reload the file.
import constants
import sys
if 'google.colab' in sys.modules:  # Testing to see if we're doing development
  import importlib
  importlib.reload(constants)

_NUMERICAL_FEATURES = constants.NUMERICAL_FEATURES
# _BUCKET_FEATURES = constants.BUCKET_FEATURES
# _FEATURE_BUCKET_COUNT = constants.FEATURE_BUCKET_COUNT
# _CATEGORICAL_NUMERICAL_FEATURES = constants.CATEGORICAL_NUMERICAL_FEATURES
_CATEGORICAL_STRING_FEATURES = constants.CATEGORICAL_STRING_FEATURES
_VOCAB_SIZE = constants.VOCAB_SIZE
_OOV_SIZE = constants.OOV_SIZE
_LABEL_KEY = constants.LABEL_KEY

#one-hot tensor to encode categorical features
#returns dense one hot tensor as flost list
#x is dense tensor
#key is string key for feature input 
def _make_one_hot(x, key):
  integerized = tft.compute_and_apply_vocabulary(x,
          top_k=_VOCAB_SIZE,
          num_oov_buckets=_OOV_SIZE,
          vocab_filename=key, name=key)
  depth = (
      tft.experimental.get_vocabulary_size_by_name(key) + _OOV_SIZE)
  one_hot_encoded = tf.one_hot(
      integerized,
      depth=tf.cast(depth, tf.int32),
      on_value=1.0,
      off_value=0.0)
  return tf.reshape(one_hot_encoded, [-1, depth])

#fill missing vals of x with ''/0
def _fill_in_missing(x):
  if not isinstance(x, tf.sparse.SparseTensor):
    return x

  default_value = '' if x.dtype == tf.string else 0
  return tf.squeeze(
      tf.sparse.to_dense(
          tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
          default_value),
      axis=1)


def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs.
  Args:
    inputs: map from feature keys to raw not-yet-transformed features.
  Returns:
    Map from string feature key to transformed feature operations.
  """
  outputs = {}
  for key in _NUMERICAL_FEATURES:
    # If sparse make it dense, setting nan's to 0 or '', and apply zscore.
    outputs[constants.t_name(key)] = tft.scale_to_z_score(
        _fill_in_missing(inputs[key]), name=key)

  #   for key in _BUCKET_FEATURES:
  #     outputs[constants.t_name(key)] = tf.cast(tft.bucketize(
  #             _fill_in_missing(inputs[key]), _FEATURE_BUCKET_COUNT, name=key),
  #             dtype=tf.float32)

    for key in _CATEGORICAL_STRING_FEATURES:
      outputs[constants.t_name(key)] = _make_one_hot(_fill_in_missing(inputs[key]), key)

  #   for key in _CATEGORICAL_NUMERICAL_FEATURES:
  #     outputs[constants.t_name(key)] = _make_one_hot(tf.strings.strip(
  #         tf.strings.as_string(_fill_in_missing(inputs[key]))), key)


  # Convert label to int
  # 0: lying, 1: running, 2: sitting, 3: walking
  outputs[constants.LABEL_KEY] = tft.compute_and_apply_vocabulary(
      _fill_in_missing(inputs[constants.LABEL_KEY]),
      top_k=None,                 # keep all classes
      num_oov_buckets=0,          # no out-of-vocab
      vocab_filename=constants.LABEL_KEY
  )
  return outputs
