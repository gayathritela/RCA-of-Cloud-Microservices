
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
file_path = '/content/drive/MyDrive/DATA/Transformed_metric_low_test/transformed_transposed_issue8_metrics.csv'
json_path = '/content/service_dependency_information (1).json'
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

def generate_comprehensive_analysis(service_name, dependencies, dependents):
    """Generate a comprehensive analysis prompt."""
    dependencies_formatted, dependents_formatted = preprocess_input(dependencies, dependents)
    llm_input1 = f"""
    Comprehensive Analysis of Service Anomalies for '{service_name}':

    Context:
    - Service: '{service_name}' is a critical part of a pet adoption website's microservices architecture.
    - Observed Anomalies: Highlighted by a high MAD score indicating significant deviations in performance metrics.

    Data Provided:
    - Dependencies: {dependencies_formatted}
    - Dependents: {dependents_formatted}

    Analysis Objectives:
    1. Root Cause Identification:
        - Evaluate the interactions within the service dependency chain.
        - Identify the likely starting point of the anomalies based on the roles and interactions detailed in the logs and summary.
        - Analyze how the identified root cause node's functionality could lead to observed issues.

    2. Impact Analysis:
        - Assess the impact on dependent services.
        - Discuss the operational and integration challenges faced by affected nodes within the overall architecture.

    Conclusion:
        - Summarize the identified root cause and most affected target node.
        - Discuss how the dependencies and dependents contribute to the propagation of issues.

    Instructions:
        - Use the provided data to support each analytical step.
        - Ensure that the analysis is logical and coherent, focusing on causality and impact within the microservices environment.
    """
    return llm_input1.strip()

def generate_detailed_impact_analysis(service_name, dependencies, dependents):
    """Generate a detailed service impact analysis prompt."""
    dependencies_formatted, dependents_formatted = preprocess_input(dependencies, dependents)
    llm_input2 = f"""
    Detailed Service Impact Analysis for '{service_name}':

    Overview:
    - Service: '{service_name}', essential to a pet adoption site, is experiencing significant performance anomalies.

    Provided Data:
    - Direct Dependencies: {dependencies_formatted}
    - Dependent Services: {dependents_formatted}

    Focus Areas:
    1. Dependencies and Their Impact:
        - Analyze the influence of '{service_name}' on its direct dependencies.
        - Assess how issues originating from '{service_name}' propagate to dependent services, affecting system performance and reliability.

    2. Pathways of Impact:
        - Map out the key pathways through which the issues are transmitted within the system.

    3. Metrics and Effects:
        - Evaluate how the issues affect critical performance metrics like latency and availability.

    4. Mitigation Strategies:
        - Propose actionable mitigation strategies to address the current issues.
        - Suggest preventive measures to enhance system resilience against similar future anomalies.

    Expected Outcomes:
        - Provide detailed insights into dependency-related impacts and propagation mechanisms.
        - Offer specific recommendations for both immediate resolution and long-term preventive strategies.

    Instructions:
        - Structure the response to ensure a logical flow, with each section addressing specific aspects as detailed above.
        - Highlight the importance of data-driven decision-making in managing microservice architectures.
    """
    return llm_input2.strip()

def analyze_root_cause(anomaly_row):
    """Analyze root cause."""
    service_name = anomaly_row['microservice']
    if not service_exists(service_name):
        return "Service info not found.", ""
    comprehensive_analysis_prompt = generate_comprehensive_analysis(service_name, service_info[service_name]['dependencies'], service_info[service_name]['dependents'])
    detailed_impact_analysis_prompt = generate_detailed_impact_analysis(service_name, service_info[service_name]['dependencies'], service_info[service_name]['dependents'])
    comprehensive_response = text_generator(comprehensive_analysis_prompt, max_new_tokens=1020, num_return_sequences=1, temperature=0.7)[0]['generated_text']
    detailed_response = text_generator(detailed_impact_analysis_prompt, max_new_tokens=1200, num_return_sequences=1, temperature=0.7)[0]['generated_text']
    cleaned_comprehensive_response = clean_hypothesis(comprehensive_response)
    cleaned_detailed_response = clean_hypothesis(detailed_response)
    return comprehensive_analysis_prompt, cleaned_comprehensive_response, detailed_impact_analysis_prompt, cleaned_detailed_response


def main():
    """Main function to analyze top services."""
    print("Calculating MAD scores...")
    top_services = calculate_mad_scores(metrics_data)
    top_services = top_services.sort_values(by='Max_MAD_Score', ascending=False).head(1)

    if top_services.empty:
        print("No services with significant MAD scores found.")
        return

    results = []
    for _, row in top_services.iterrows():
        print(f"Analyzing root cause for service: {row['microservice']}...")
        comprehensive_prompt, comprehensive_response, detailed_prompt, detailed_response = analyze_root_cause(row)
        results.append({
            'Service': row['microservice'],
            'Timestamp': row['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
            'MAD Score': row['Max_MAD_Score'],
            'Affected Metric': row['Metric_With_Max_MAD'],
            'Comprehensive Analysis Prompt': comprehensive_prompt,
            'Comprehensive Hypothesis': comprehensive_response,
            'Detailed Impact Analysis Prompt': detailed_prompt,
            'Detailed Impact Hypothesis': detailed_response
        })
    results_df = pd.DataFrame(results)
    results_file_path = '/content/Low_Traffic_Test_issue11_WithoutRAG.csv'
    results_df.to_csv(results_file_path, index=False)
    print(f"Results saved to {results_file_path}.")


if _name_ == "_main_":
    main()
