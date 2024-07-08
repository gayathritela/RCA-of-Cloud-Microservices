import pandas as pd
import numpy as np
import json
import os
import random
from datetime import datetime
from sklearn.ensemble import IsolationForest
import zipfile
from sentence_transformers import SentenceTransformer
import pinecone

def load_data(file_path):
    """ Load data from a CSV file. """
    return pd.read_csv(file_path)

def generate_synthetic_data(df, column_ranges, noise_level=0.1, tolerance_factor=0.3):
    """ Generate synthetic data with more variability. """
    synthetic_df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        range_width = column_ranges[col]['max'] - column_ranges[col]['min']
        noise = np.random.normal(0, range_width * noise_level, size=len(df))
        synthetic_df[col] = df[col] + noise
        synthetic_df[col] = np.clip(synthetic_df[col], column_ranges[col]['min'], column_ranges[col]['max'])
    return synthetic_df

def dynamic_severity_adjustment(base_severity):
    """ Adjust severity based on the current time of day. """
    current_hour = datetime.now().hour
    adjusted_severity = base_severity * 1.2 if 8 <= current_hour <= 20 else base_severity * 0.8
    return adjusted_severity + random.uniform(-0.05, 0.05)

def inject_issue(df, path_set, column_ranges, issue_type='latency', base_severity=0.1):
    """ Inject an issue into both root cause node and target node and mark them accordingly in the issue_injected column.
        Additionally, include the path involved in each row. """
    suitable_columns = [col for col in df.columns if col.endswith('Average') or col.startswith('latency_')]
    issue_column = random.choice(suitable_columns)
    path = random.choice(path_set)
    microservice = random.choice(path)  # This will be considered as the root cause node.
    target_node = random.choice([node for node in path if node != microservice])  # Target node.

    severity = dynamic_severity_adjustment(base_severity)
    adjustment_factor = 1 + severity if issue_type == 'latency' else 1 - severity

    df.loc[:, issue_column] *= adjustment_factor
    df.loc[:, issue_column] = np.clip(df.loc[:, issue_column], column_ranges[issue_column]['min'], column_ranges[issue_column]['max'])

    df['issue_injected'] = df['microservice'].apply(
        lambda x: 'root_cause_node' if x == microservice else ('target_node' if x == target_node else 'no_issue'))

    path_str = ' -> '.join(path)
    df['path'] = path_str

    return df, {'target_node': target_node, 'root_cause_node': microservice, 'issue_type': issue_type, 'severity': severity, 'affected_column': issue_column, 'timestamp': str(datetime.now()), 'paths': path}

def save_data(df, ground_truth, output_dir, i):
    """ Save synthetic data and corresponding ground truth. """
    synthetic_file_path = os.path.join(output_dir, f'synthetic_data_{i}_issue.csv')
    ground_truth_file_path = os.path.join(output_dir, f'ground_truth_{i}_issue.json')
    df.to_csv(synthetic_file_path, index=False)
    with open(ground_truth_file_path, 'w') as f:
        json.dump(ground_truth, f)

def apply_isolation_forest(df, numeric_columns):
    """ Detect anomalies using the Isolation Forest algorithm. """
    isolation_forest = IsolationForest(contamination=0.2)
    isolation_forest.fit(df[numeric_columns])
    df['anomaly'] = isolation_forest.predict(df[numeric_columns])
    return df[df['anomaly'] == -1]

def evaluate_anomalies(df, ground_truth, numeric_columns):
    """ Evaluate detected anomalies against ground truth. """
    anomalies = apply_isolation_forest(df, numeric_columns)
    anomaly_indices = set(anomalies.index)
    injected_indices = set(df[df['issue_injected'] != 'no_issue'].index)
    correctly_detected = anomaly_indices.intersection(injected_indices)
    false_negatives = injected_indices.difference(anomaly_indices)
    false_positives = anomaly_indices.difference(injected_indices)

    evaluation_summary = {
        'correctly_detected': len(correctly_detected),
        'false_negatives': len(false_negatives),
        'false_positives': len(false_positives),
        'total_injected': len(injected_indices),
        'total_detected': len(anomaly_indices),
        'details': {
            'correctly_detected_indices': list(correctly_detected),
            'false_negative_indices': list(false_negatives),
            'false_positive_indices': list(false_positives)
        }
    }
    return evaluation_summary

def create_textual_representation(row):
    text = f"On {row['timestamp']}, the service {row['microservice']} reported metrics with "
    text += f"an availability of {row['availability_Average']}%, "
    text += f"an average latency of {row['latency_Average']} ms, "
    text += f"and peak latencies at p50: {row['latency_p50']} ms, p90: {row['latency_p90']} ms, p95: {row['latency_p95']} ms, "
    text += f"and p99: {row['latency_p99']} ms. Total requests were {row['requests_Sum']}. "

    if row['issue_injected'] == 'root_cause_node':
        text += f"This service was identified as the root cause of a performance issue. "
    elif row['issue_injected'] == 'target_node':
        text += f"This service was affected as a target node in the performance issue. "
    else:
        text += f"No significant issues were detected. "

    text += f"Service path involved: {row['path']}."

    return text


def main(path_set_file, original_file_paths, output_dir, num_datasets, column_ranges):
    """ Main function to process data and evaluate anomalies. """
    with open(path_set_file) as f:
        path_set = json.load(f)

    evaluations = []
    os.makedirs(output_dir, exist_ok=True)

    for i, file_path in enumerate(original_file_paths):
        df = load_data(file_path)
        for j in range(num_datasets):
            synthetic_df = generate_synthetic_data(df, column_ranges)
            synthetic_df, ground_truth = inject_issue(synthetic_df, path_set, column_ranges)

            synthetic_df['textual_representation'] = synthetic_df.apply(create_textual_representation, axis=1)

            save_data(synthetic_df, ground_truth, output_dir, f'{i}_{j}')

            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            evaluation = evaluate_anomalies(synthetic_df, ground_truth, numeric_columns)
            evaluations.append(evaluation)
            print(f"Evaluation for dataset {i}_{j}: {evaluation}")

    save_evaluation_results(evaluations, output_dir)
    zip_synthetic_data(output_dir, 'synthetic_data.zip')

def save_evaluation_results(evaluations, output_dir):
    """ Save a summary of the evaluations. """
    eval_df = pd.DataFrame(evaluations)
    eval_summary_path = os.path.join(output_dir, 'evaluations_summary.csv')
    eval_df.to_csv(eval_summary_path, index=False)
    print(eval_df)

def zip_synthetic_data(output_dir, zip_filename):
    """ Zip all synthetic data files into a single zip file. """
    with zipfile.ZipFile(os.path.join(output_dir, zip_filename), 'w') as zipf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                if file.endswith('.csv') or file.endswith('.json'):
                    zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), output_dir))

if __name__ == "__main__":
    path_set_file = '/content/path_sets.json'
    original_file_paths = [
        '/content/transformed_transposed_issue0_metrics.csv',
        '/content/transformed_transposed_issue1_metrics.csv',
        '/content/transformed_transposed_issue2_metrics.csv'
    ]
    output_dir = '/content/Synthetic_datasets_v1'
    column_ranges = {
        'availability_Average': {'min': 0.0, 'max': 100.0},
        'latency_Average': {'min': 0.0, 'max': 4.91},
        'latency_p50': {'min': 0.0, 'max': 5.83},
        'latency_p90': {'min': 0.0, 'max': 6.60},
        'latency_p95': {'min': 0.0, 'max': 6.74},
        'latency_p99': {'min': 0.0, 'max': 6.76},
        'requests_Sum': {'min': 0.0, 'max': 1287.0}
    }
    main(path_set_file, original_file_paths, output_dir, 50, column_ranges)

     
