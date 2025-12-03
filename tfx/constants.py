NUMERIC_FEATURE_KEYS = ['ax','ay', 'az'] # timestamp is only used for windowing
CATEGORICAL_FEATURE_KEYS = ['participant_id']
LABEL_KEY = 'activity_id'

def transformed_name(key):
    return key + '_xf'