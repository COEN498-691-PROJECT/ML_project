
NUMERICAL_FEATURES = ['ax_mean','ax_std','ax_max','ax_min','ax_range',
            'ax_skew','ax_kurt','ax_zcr',
            'ay_mean','ay_std','ay_max','ay_min','ay_range',
            'ay_skew','ay_kurt','ay_zcr',
            'az_mean','az_std','az_max','az_min','az_range',
            'az_skew','az_kurt','az_zcr',
            'sma','corr_axy','corr_axz','corr_ayz',
            'axG_mean','ayG_mean','azG_mean',
            'Gx','Gy','Gz','Gx_angle','Gy_angle','Gz_angle'
]

# BUCKET_FEATURES = [
#     ...
# ]
# # Number of buckets used by tf.transform for encoding each feature.
# FEATURE_BUCKET_COUNT = 10

CATEGORICAL_STRING_FEATURES = [
    'participant_id'
]

# Number of vocabulary terms used for encoding categorical features.
VOCAB_SIZE = 1000

# Count of out-of-vocab buckets in which unrecognized categorical are hashed.
# no bucket
OOV_SIZE = 0

# Keys
LABEL_KEY = 'activity_id'


def t_name(key):
  """
  Rename the feature keys so that they don't clash with the raw keys when
  running the Evaluator component.
  Args:
    key: The original feature key
  Returns:
    key with '_xf' appended
  """
  return key + '_xf'
