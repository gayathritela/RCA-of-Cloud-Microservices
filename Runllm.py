import pandas as pd
from keybert import KeyBERT
import json
from scipy.stats import median_abs_deviation
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer, util


# Load Sentence Transformers
model_semantic = SentenceTransformer('all-MiniLM-L6-v2')
model_scoring = SentenceTransformer('bert-base-nli-mean-tokens')

#Load huggingface models
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Genstruct-7B")
model = AutoModelForCausalLM.from_pretrained("NousResearch/Genstruct-7B")
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, truncation=True)


# Load the dataset and service information
file_path = '/content/transformed_transposed_issue2_metrics.csv'
json_path = '/content/updated_injected_issues_summary (2).json'
metrics_data = pd.read_csv(file_path)


# Conversion 'timestamp_' column to datetime
metrics_data['timestamp'] = pd.to_datetime(metrics_data['timestamp'])

with open(json_path, 'r') as f:
    data = json.load(f)
    service_info = {service['name']: {'dependencies': service['dependencies'], 'dependents': service['dependents']} for service in data['services']}

def service_exists(service_name):
    """Check if a service exists in the loaded service information."""
    return service_name in service_info

# Define reference texts for quality evaluation
reference_texts = [
    "The service disruption was primarily caused by an unexpected surge in traffic, leading to database bottlenecks.",
    "A memory leak in the service's caching layer resulted in gradual degradation of response times.",
    "Due to inadequate load balancing, the increased load was not properly distributed across instances, causing some to be overwhelmed."
]

def calculate_mad_scores(dataframe):
    """Calculate MAD scores for metrics and select the top services with the highest scores."""
    # Adjust to match the column names from the loaded Excel file
    metric_columns = ['availability_Average', 'latency_Average', 'latency_p50', 'latency_p90', 'latency_p95', 'latency_p99', 'requests_Sum']
    # Ensure these columns exist in the dataframe to avoid KeyError
    missing_columns = [col for col in metric_columns if col not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in DataFrame: {missing_columns}")

    mad_columns = [f'{metric}_MAD' for metric in metric_columns]
    # Group by the service column name and calculate MAD
    for metric in metric_columns:
        dataframe[metric + '_MAD'] = dataframe.groupby('microservice')[metric].transform(lambda x: median_abs_deviation(x, scale='normal'))

    dataframe['Max_MAD_Score'] = dataframe[mad_columns].max(axis=1)
    dataframe['Metric_With_Max_MAD'] = dataframe[mad_columns].idxmax(axis=1).str.replace('_MAD', '')
    return dataframe




def evaluate_response_quality(hypothesis, reference_texts):
    hypothesis_embedding = model_scoring.encode(hypothesis, convert_to_tensor=True)
    reference_embeddings = model_scoring.encode(reference_texts, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(hypothesis_embedding, reference_embeddings)

    # Return the maximum similarity score as the quality score
    return cosine_scores.max().item() * 100  # Converted to percentage

def refine_hypothesis(prompt, reference_texts, iterations=3, threshold=80):
    for i in range(iterations):
        response = text_generator(prompt, max_length=1200, num_return_sequences=1, temperature=0.7, top_p=0.9)[0]['generated_text']
        cleaned_response = clean_hypothesis(response)
        score = evaluate_response_quality(cleaned_response, reference_texts)

        if score > threshold:
            return cleaned_response, score  # Return the response and its score if it's above the threshold

        prompt = rephrase_prompt(prompt, cleaned_response)  # Refine the prompt based on the response

    return cleaned_response, score  # Return the last response and score if none meet the threshold


def rephrase_prompt(original_prompt, response):
    """Modify the prompt based on the feedback from the previous response."""
    # Example adjustment: Ask for more details or clarity if the response is vague
    return original_prompt + "\n Please refine your last response for more accuracy and detail."

def clean_hypothesis(hypothesis):
    """Process the hypothesis to refine and extract the relevant section."""
    lines = hypothesis.split('\n')
    # Focus on extracting content after the last 'Step' to get the final hypothesis
    final_hypothesis_start = next((i for i, line in enumerate(lines) if "Final Hypothesis:" in line), None)
    return '\n'.join(lines[final_hypothesis_start + 1:]) if final_hypothesis_start is not None else hypothesis

def score_hypothesis_semantic(hypothesis, reference_texts):
    hypothesis_embedding = model_semantic.encode(hypothesis, convert_to_tensor=True)
    reference_embeddings = model_semantic.encode(reference_texts, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(hypothesis_embedding, reference_embeddings)

    return cosine_scores.max().item() * 100  # Convert to percentage
# This parameter can be adjusted according to the desired quality of the output.
threshold = 80

def preprocess_input(dependencies, dependents):
    """Preprocess input to ensure clear, structured prompts."""
    dependencies_formatted = ', '.join(dependencies) if dependencies else 'None'
    dependents_formatted = ', '.join(dependents) if dependents else 'None'
    return dependencies_formatted, dependents_formatted

def generate_analysis_prompt(service_name, mad_score, affected_metric, dependencies, dependents):
    """Generate a structured analysis prompt for the model."""
    dependencies_formatted, dependents_formatted = preprocess_input(dependencies, dependents)
    prompt = f"""
    An anomaly with a Median Absolute Deviation (MAD) score of {mad_score} has been detected in the {service_name} service's {affected_metric} metric, indicating a substantial deviation impacting its performance. This service is a critical component of a pet adoption website's microservices architecture.

    Dependencies involved include: {dependencies_formatted}.
    The service also serves as a crucial dependency for: {dependents_formatted}.

    Your analysis should focus on identifying a singular root cause from among the dependencies. This cause should directly contribute to the anomaly in the {affected_metric} metric. Consider each dependency's role and potential issues that could lead to such a deviation.

    Additionally, pinpoint the primary dependent (target node) that is most directly affected by this anomaly. This should be the service that relies on {service_name} and would face the most significant impact due to the anomaly in {affected_metric}.

    Please provide a concise and focused hypothesis on:
    1. The singular root cause node among the dependencies and dependents.
    2. The primary target node among the dependents directly impacted by this anomaly.

    Your analysis will guide subsequent investigation and mitigation efforts.
    """

    print("Generated prompt:\n", prompt)
    return prompt.strip()


def analyze_root_cause(anomaly_row):
    service_name = anomaly_row['microservice']
    if not service_exists(service_name):
        return "Service information not found.", "", []

    prompt = generate_analysis_prompt(
        service_name,
        anomaly_row['Max_MAD_Score'],
        anomaly_row['Metric_With_Max_MAD'],
        service_info[service_name]['dependencies'],
        service_info[service_name]['dependents']
    )

    response = text_generator(prompt, max_length=1200, num_return_sequences=1, temperature=0.7, top_p=0.9)[0]['generated_text']
    cleaned_response = clean_hypothesis(response)
    keywords = extract_keywords(cleaned_response)
    return prompt, cleaned_response, keywords



def extract_keywords(text):
    kw_model = KeyBERT(model='distilbert-base-nli-mean-tokens')
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
    return [kw[0] for kw in keywords]  # Return only keywords, not scores

def main():
    print("Calculating adaptive MAD scores...")
    global metrics_data  # Use the global metrics_data variable
    metrics_data = calculate_mad_scores(metrics_data)  # Update your metrics_data DataFrame
    top_services = metrics_data.sort_values(by='Max_MAD_Score', ascending=False).head(2)  # Get the top 3 services
    results = []

    for index, row in top_services.iterrows():
        print(f"Refining analysis for service: {row['microservice']}...")
        prompt = generate_analysis_prompt(
            row['microservice'], row['Max_MAD_Score'], row['Metric_With_Max_MAD'],
            service_info[row['microservice']]['dependencies'], service_info[row['microservice']]['dependents'])
        refined_response, _ = refine_hypothesis(prompt, reference_texts)  # Ignore the evaluation score
        keywords = extract_keywords(refined_response)
        results.append({
        'Service': row['microservice'],
        'Timestamp': row['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
        'MAD Score': row['Max_MAD_Score'],
        'Affected Metric': row['Metric_With_Max_MAD'],
        'Prompt': prompt,
        'Hypothesis': refined_response,
        })

    # Convert results list to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv('/content/LOWTestissue2gen.csv', index=False)
    print("Analysis results saved.")

main()