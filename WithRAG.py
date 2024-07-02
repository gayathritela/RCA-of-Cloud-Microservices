import json
import pandas as pd
from scipy.stats import median_abs_deviation
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_community.graphs import Neo4jGraph

# Initialize Hugging Face Pipeline
hf = HuggingFacePipeline.from_model_id(
    model_id="NousResearch/Hermes-2-Pro-Mistral-7B",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 1050, "temperature": 0.7, "top_p": 0.9, "top_k": 50}
)

# Initialize Neo4j Graph connection
graph = Neo4jGraph(url="neo4j+s://ffd05695.databases.neo4j.io", username="neo4j", password="RMGf-ZEmdSzk766LJX_tZrHY_X1j_H7qqoXUsaDPOcA")

# Load data
file_path = '/content/drive/MyDrive/DATA/Transformed_metric_high_test/transformed_transposed_issue8_metrics.csv'
metrics_data = pd.read_csv(file_path)
metrics_data['timestamp'] = pd.to_datetime(metrics_data['timestamp'])
print("Data loaded and timestamp converted.")

def calculate_mad_scores(dataframe):
    metrics = ['availability_Average', 'latency_Average', 'latency_p50', 'latency_p90', 'latency_p95', 'latency_p99', 'requests_Sum']
    for metric in metrics:
        dataframe[metric + '_MAD'] = dataframe.groupby('microservice')[metric].transform(lambda x: median_abs_deviation(x, scale='normal'))
    dataframe['Max_MAD_Score'] = dataframe[[metric + '_MAD' for metric in metrics]].max(axis=1)
    top_service = dataframe.loc[dataframe['Max_MAD_Score'].idxmax()]
    print(f"Calculated MAD Scores. Top service: {top_service['microservice']} with score: {top_service['Max_MAD_Score']}")
    return top_service['microservice'], top_service['Max_MAD_Score']


def generate_response(llm_input, max_tokens, temperature=0.7, top_p=0.9, top_k=50):
    response = hf(llm_input, max_new_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k)
    return response



def run_query_and_generate_response(top_service,max_tokens1, max_tokens2):
    microservice_id = top_service['service_id']
    service_name = top_service['name']

    query_info = """
      MATCH (m:Microservice)-[r:CAUSES]->(i:Issue)
      WHERE toLower(m.service_id) = toLower($microservice_id) AND m.root_cause_node = TRUE
      RETURN m.textual_representation AS MicroserviceText, i.textual_representation AS IssueSummary;
      """

    query_dependencies = """
    MATCH path=(dependent:Service)-[:USED_BY*]->(service:Service {name: $service_name})
    RETURN collect(distinct dependent.name) AS Dependencies
    """

    query_dependents = """
    MATCH path=(root:Service {name: $service_name})-[:DEPENDS_ON*]->(dependent)
    RETURN collect(distinct dependent.name) AS Dependents
    """

    params_info = {"microservice_id": microservice_id}
    params_deps = {"service_name": service_name}

    results_info = graph.query(query_info, params_info)
    dependencies = graph.query(query_dependencies, params_deps)
    dependents = graph.query(query_dependents, params_deps)

    dependencies_list = ', '.join(dependencies[0]['Dependencies']) if dependencies and 'Dependencies' in dependencies[0] else "No dependencies listed."
    dependents_list = ', '.join(dependents[0]['Dependents']) if dependents and 'Dependents' in dependents[0] else "No dependents listed."

    if results_info:
        result = results_info[0]
        print("Fetched Service Information:")
        print(f"Microservice Text: {result['MicroserviceText']}")
        print(f"Issue Summary: {result['IssueSummary']}")
        print(f"Dependencies: {dependencies_list}")
        print(f"Dependents: {dependents_list}")

        proceed = input("Proceed with generating the response? (yes/no): ")
        if proceed.lower() != 'yes':
            return ("Aborted by user.", None), ("Aborted by user.", None)

        llm_input1 = f"""
    Comprehensive Analysis of Service Anomalies for '{service_name}':

    Context:
    - Service: '{service_name}' is a critical part of a pet adoption website's microservices architecture.
    - Observed Anomalies: Highlighted by a high MAD score indicating significant deviations in performance metrics.

    Data Provided:
    - Service Logs: {result['MicroserviceText']}
    - Issue Summary: {result['IssueSummary']}
    - Dependencies: {dependencies_list}
    - Dependents: {dependents_list}

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
        - Ensure that the analysis is logical and coherent, focusing on causality and impact within the microservices environment."""

        llm_input2 = f"""
    Detailed Service Impact Analysis for '{service_name}':

    Overview:
    - Service: '{service_name}', essential to a pet adoption site, is experiencing significant performance anomalies.

    Provided Data:
    - Performance Data: {result['MicroserviceText']}
    - Issues Identified: {result['IssueSummary']}
    - Direct Dependencies: {dependencies_list}
    - Dependent Services: {dependents_list}

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


        response1 = generate_response(llm_input1, max_tokens1)
        response2 = generate_response(llm_input2, max_tokens2)



        print(f"Response generated for service: {service_name}")
        return (response1, llm_input1), (response2, llm_input2)
    else:
        print("No data found for the given service ID.")
        return ("No data found for the given service ID.", None), (None, None)

def main():
    top_service_name, top_mad_score = calculate_mad_scores(metrics_data)
    top_service = {'service_id': top_service_name, 'name': top_service_name}

    max_tokens1 = 1000 # Adjust these based on your needs
    max_tokens2 = 1050   # Adjust these based on your needs

    (response1, llm_input1), (response2, llm_input2) = run_query_and_generate_response(top_service,  max_tokens1, max_tokens2)

    if llm_input1 and llm_input2:
        data = {
            "Service Name": [top_service_name, top_service_name],
            "MAD Score": [top_mad_score, top_mad_score],
            "Prompt": [llm_input1, llm_input2],
            "Response": [response1, response2]
        }

        df = pd.DataFrame(data)
        csv_file_path = '/content/hightraffictestissue8v1.csv'
        df.to_csv(csv_file_path, index=False)
        print(f"Data successfully saved to {csv_file_path}")
    else:
        print("No input generated or user aborted the operation.")

if __name__ == "__main__":
    main()

