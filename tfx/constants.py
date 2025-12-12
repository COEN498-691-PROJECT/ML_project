
NUMERICAL_FEATURES = [
    'ax',
    'ay',
    'az', 
    'timestamp'
]

CATEGORICAL_STRING_FEATURES = [
    'participant_id'
]

# Number of vocabulary terms used for encoding categorical features.
VOCAB_SIZE = 1000

# Count of out-of-vocab buckets in which unrecognized categorical are hashed.
# no bucket so set to 0
OOV_SIZE = 0

# Keys
LABEL_KEY = 'activity_id'

#rename feature keys to prevent clash with raw keys names
def t_name(key):
  return key + '_xf'
