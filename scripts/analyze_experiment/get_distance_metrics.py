import numpy as np
import pandas as pd
import re

# Load the files
cosine_scores = np.load('../../data/experiment_data/cosine_scores.npy')
embeddings = np.load('../../data/experiment_data/embeddings.npy')
idx2rid = np.load('../../data/experiment_data/idx2rid.npy', allow_pickle=True).item()
rid2idx = np.load('../../data/experiment_data/rid2idx.npy', allow_pickle=True).item()
expr_data = pd.read_csv('../../data/experiment_data/expr_data.csv')

# Convert `init_array` column from string to list
expr_data['init_array'] = expr_data['init_array'].apply(lambda x: re.findall(r"'(.*?)'", x))
expr_data['response_id'] = expr_data['response_id'].str.replace(r'_world\d+$', '', regex=True)
expr_data['init_array'] = expr_data['init_array'].apply(lambda ids: [re.sub(r'_world\d+$', '', id) for id in ids])


def calculate_distance_metrics(row, rid2idx, cosine_scores):
    response_id = row['response_id']
    init_array = row['init_array']

    if response_id not in rid2idx or not init_array:
        return np.nan, np.nan, np.nan, np.nan, {}

    response_idx = rid2idx[response_id]
    init_indices = [rid2idx[rid] for rid in init_array if rid in rid2idx]

    # Get the cosine scores between response_id and each element in init_array
    scores = cosine_scores[response_idx, init_indices]
    # Convert cosine similarity to cosine distance
    distances = 1 - scores

    # Calculate average distances
    avg_distance = np.mean(distances)

    # Split the init_array into arrays starting with 'ai' and not starting with 'ai'
    ai_init_indices = [idx for idx, rid in zip(init_indices, init_array) if rid.startswith('ai')]
    non_ai_init_indices = [idx for idx, rid in zip(init_indices, init_array) if not rid.startswith('ai')]

    # Get the distances for 'ai' and non-'ai' arrays
    ai_distances = 1 - cosine_scores[response_idx, ai_init_indices]
    non_ai_distances = 1 - cosine_scores[response_idx, non_ai_init_indices]

    # Calculate average distances
    avg_ai_distance = np.mean(ai_distances) if len(ai_distances) > 0 else np.nan
    avg_non_ai_distance = np.mean(non_ai_distances) if len(non_ai_distances) > 0 else np.nan

    # Calculate average distance for the entire set
    total_avg_distance = np.mean(np.append(distances, 1 - cosine_scores[response_idx, response_idx]))

    # Calculate distance metrics
    metrics = {
        'mean': np.mean,
        'median': np.median,
        'sd': np.std,
        'min': np.min,
        'max': np.max,
        'sum': np.sum
    }

    distance_metrics = {}
    for key, func in metrics.items():
        distance_metrics[f'{key}_init_array'] = func(distances) if len(distances) > 0 else np.nan
        distance_metrics[f'{key}_ai'] = func(ai_distances) if len(ai_distances) > 0 else np.nan
        distance_metrics[f'{key}_non_ai'] = func(non_ai_distances) if len(non_ai_distances) > 0 else np.nan
        distance_metrics[f'{key}_all'] = func(np.append(distances, 1 - cosine_scores[response_idx, response_idx]))

    return avg_distance, avg_ai_distance, avg_non_ai_distance, total_avg_distance, distance_metrics


def main():
    # Apply the function to the DataFrame
    df = expr_data.dropna(subset=['response_text'])
    result = df.apply(lambda row: calculate_distance_metrics(row, rid2idx, cosine_scores), axis=1, result_type='expand')
    df[['avg_pw_dist', 'avg_ai_pw_dist', 'avg_non_ai_pw_dist', 'total_avg_pw_dist', 'distance_metrics']] = result

    # Expand the distance metrics dictionary into separate columns
    df = pd.concat([df, df['distance_metrics'].apply(pd.Series)], axis=1)
    df['ai_diff'] = df['avg_ai_pw_dist'] - df['avg_non_ai_pw_dist']
    df = df[['response_id', 'avg_pw_dist', 'avg_ai_pw_dist', 'avg_non_ai_pw_dist',
             'total_avg_pw_dist', 'mean_init_array', 'mean_ai',
             'mean_non_ai', 'mean_all', 'median_init_array', 'median_ai',
             'median_non_ai', 'median_all', 'sd_init_array', 'sd_ai', 'sd_non_ai',
             'sd_all', 'min_init_array', 'min_ai', 'min_non_ai', 'min_all',
             'max_init_array', 'max_ai', 'max_non_ai', 'max_all', 'sum_init_array',
             'sum_ai', 'sum_non_ai', 'sum_all', 'ai_diff']]
    df.to_csv('../../data/experiment_data/distance_metrics.csv', index=False)


if __name__ == '__main__':
    main()