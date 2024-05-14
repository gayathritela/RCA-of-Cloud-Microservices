import pandas as pd
import json
from scipy.stats import median_abs_deviation
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load models
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Hermes-2-Pro-Mistral-7B")
model = AutoModelForCausalLM.from_pretrained("NousResearch/Hermes-2-Pro-Mistral-7B")
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, truncation=True)

print("Models loaded.")

# Load data
file_path = '/content/transformed_transposed_issue1_metrics.csv'
json_path = '/content/service_dependency_information.json'
metrics_data = pd.read_csv(file_path)
metrics_data['timestamp'] = pd.to_datetime(metrics_data['timestamp'])

# Load service info
with open(json_path, 'r') as f:
    data = json.load(f)
    service_info = {s['name']: {'dependencies': s['dependencies'], 'dependents': s['dependents']} for s in data['services']}

def service_exists(service_name):
    """Check if service exists."""
    return service_name in service_info

def calculate_mad_scores(dataframe):
    """Calculate MAD scores."""
    metrics = ['availability_Average', 'latency_Average', 'latency_p50', 'latency_p90', 'latency_p95', 'latency_p99', 'requests_Sum']
    mad_columns = [m + '_MAD' for m in metrics]
    for m in metrics:
        dataframe[m + '_MAD'] = dataframe.groupby('microservice')[m].transform(lambda x: median_abs_deviation(x, scale='normal'))
    dataframe['Max_MAD_Score'] = dataframe[mad_columns].max(axis=1)
    dataframe['Metric_With_Max_MAD'] = dataframe[mad_columns].idxmax(axis=1).str.replace('_MAD', '')
    return dataframe.loc[dataframe.groupby('microservice')['Max_MAD_Score'].idxmax()]

def preprocess_input(dependencies, dependents):
    dependencies_formatted = ', '.join(dependencies) if dependencies else 'No direct dependencies; focus on dependents or the anomaly itself.'
    dependents_formatted = ', '.join(dependents) if dependents else 'None'
    return dependencies_formatted, dependents_formatted

def clean_hypothesis(hypothesis):
    lines = hypothesis.split('\n')
    seen = set()
    unique_lines = [line for line in lines if line and line not in seen and not seen.add(line)]
    return '\n'.join(unique_lines)

def generate_analysis_prompt(service_name, mad_score, affected_metric, dependencies, dependents):
    dependencies_formatted, dependents_formatted = preprocess_input(dependencies, dependents)
    prompt = f"""
    An anomaly with a Median Absolute Deviation (MAD) score of {mad_score} has been detected in the {service_name} service's {affected_metric} metric, indicating a substantial deviation impacting its performance. This service is a critical component of a pet adoption website's microservices architecture.

     The service relies on the following dependencies: {dependencies_formatted}.
    The service also serves as a crucial dependency for: {dependents_formatted}.

    Your analysis should focus on identifying a singular root cause from among the dependencies and dependents.  Consider each dependency's role and potential issues that could lead to such a deviation.

    Additionally, pinpoint the primary dependent (target node) that is most directly affected by this anomaly. This should be the service that relies on {service_name} and would face the most significant impact due to the anomaly in {affected_metric}.

    Please provide a concise and focused hypothesis on:
    1. The singular root cause node among the dependencies and dependents.
    2. The primary target node among the dependents directly impacted by this anomaly.

    Your analysis will guide subsequent investigation and mitigation efforts.
    """

    print("Generated prompt:\n", prompt)
    return prompt.strip()

def analyze_root_cause(anomaly_row):
    """Analyze root cause."""
    service_name = anomaly_row['microservice']
    if not service_exists(service_name):
        return "Service info not found.", ""
    prompt = generate_analysis_prompt(
        service_name,
        anomaly_row['Max_MAD_Score'],
        anomaly_row['Metric_With_Max_MAD'],
        service_info[service_name]['dependencies'],
        service_info[service_name]['dependents']
    )
    response = text_generator(prompt, max_new_tokens=1000, num_return_sequences=1, temperature=0.7)[0]['generated_text']
    return prompt, clean_hypothesis(response)

def main():
    """Main function to analyze top services."""
    print("Calculating MAD scores...")
    top_services = calculate_mad_scores(metrics_data)
    top_services = top_services.sort_values(by='Max_MAD_Score', ascending=False).head(2)
    results = []
    for _, row in top_services.iterrows():
        print(f"Analyzing root cause for service: {row['microservice']}...")
        prompt, cleaned_response = analyze_root_cause(row)
        results.append({
            'Service': row['microservice'],
            'Timestamp': row['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
            'MAD Score': row['Max_MAD_Score'],
            'Affected Metric': row['Metric_With_Max_MAD'],
            'Prompt': prompt,
            'Hypothesis': cleaned_response
        })
    results_df = pd.DataFrame(results)
    results_file_path = '/content/temp2trainissue1mist.csv'
    results_df.to_csv(results_file_path, index=False)
    print(f"Results saved to {results_file_path}.")

if __name__ == "__main__":
    main()
