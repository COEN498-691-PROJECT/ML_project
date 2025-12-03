import tensorflow as tf
import tensorflow_transform as tft
import constants

_NUMERIC_FEATURE_KEYS = constants.NUMERIC_FEATURE_KEYS
_CATEGORICAL_FEATURE_KEYS = constants.CATEGORICAL_FEATURE_KEYS
_LABEL_KEY = constants.LABEL_KEY
_transformed_name = constants.transformed_name

def preprocessing_fn(inputs):

    outputs = {}

    # Perform per-channel normalization
    for key in _NUMERIC_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.scale_to_z_score(inputs[key])

    # Segment the data into overlapping windows of 1s and 50% overlap


    # Encode categorical features
    for key in _CATEGORICAL_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.compute_and_apply_vocabulary(inputs[key])

    # Convert the label to integer indices
    outputs[_transformed_name(_LABEL_KEY)] = tft.compute_and_apply_vocabulary(inputs[_LABEL_KEY])

    return outputs