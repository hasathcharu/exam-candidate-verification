import numpy as np
import pandas as pd

excluded = ['sample', 'and_presence', 'the_presence', 'and_latent_0', 'and_latent_1', 'and_latent_2', 'and_latent_3', 'and_latent_4', 'and_latent_5', 'the_latent_0', 'the_latent_1', 'the_latent_2', 'the_latent_3', 'the_latent_4', 'the_latent_5']


def add_gaussian_noise(features, mean=0, std=0.05):
    original_order = features.columns
    valid_excluded = [col for col in excluded if col in features.columns]
    excluded_feat = features[valid_excluded]
    feature_values = features.drop(columns=valid_excluded).astype(float)
    noise = np.random.normal(mean, std, feature_values.shape)
    augmented = feature_values + noise
    augmented = augmented.clip(lower=0)
    augmented = pd.concat([excluded_feat, augmented], axis=1)
    augmented = augmented[original_order]
    return augmented

def random_feature_scaling(features, scale_range=(0.9, 1.1)):
    original_order = features.columns
    valid_excluded = [col for col in excluded if col in features.columns]
    excluded_feat = features[valid_excluded]
    feature_values = features.drop(columns=valid_excluded).astype(float)
    scale_factors = np.random.uniform(*scale_range, feature_values.shape[1])
    scaled = feature_values * scale_factors
    scaled = pd.concat([excluded_feat, scaled], axis=1)
    scaled = scaled[original_order]
    return scaled


# def interpolate_features(features):
#     ids = features[['sample']]
#     feature_values = features.drop(columns=['sample'] + excluded).astype(float)

#     new_samples = []
#     new_ids = []

#     for _ in range(len(features) // 2):
#         idx1, idx2 = np.random.choice(len(features), 2, replace=False)
#         alpha = np.random.rand()
#         new_sample = alpha * feature_values.iloc[idx1] + (1 - alpha) * feature_values.iloc[idx2]
#         new_samples.append(new_sample)
#         new_ids.append(ids.iloc[idx1])

#     new_samples = pd.DataFrame(new_samples, columns=feature_values.columns)
#     new_ids = pd.DataFrame(new_ids).reset_index(drop=True)

#     return pd.concat([new_ids, new_samples], axis=1)

def feature_dropout(features, dropout_rate=0.1):
    original_order = features.columns
    valid_excluded = [col for col in excluded if col in features.columns]
    excluded_feat = features[valid_excluded]
    feature_values = features.drop(columns=valid_excluded).astype(float)
    
    mask = np.random.rand(*feature_values.shape) < dropout_rate
    dropped = feature_values.mask(mask, other=feature_values.mean(), axis=1)
    dropped = pd.concat([excluded_feat, dropped], axis=1)
    dropped = dropped[original_order]
    return dropped

def augment_data(data):
    return pd.concat(
        [
            df
            for df in [
                data,
                add_gaussian_noise(data),
                random_feature_scaling(data),
                # interpolate_features(data),
                feature_dropout(data),
            ]
            if not df.empty
        ],
        ignore_index=True,
    )
