
# Root Cause Analysis of Cloud Microservices

## 🌐 Project Overview



This project focuses on analyzing root causes of anomalies in cloud microservices using both traditional models and Large Language Models (LLMs). The primary dataset used is the "PetShop Dataset," which is designed to facilitate root cause analysis in microservice environments.

### 🔑 Key Features

- **🔄 Data Preprocessing**: Transform and prepare data for analysis.
- **🚨 Anomaly Detection**: Use Median Absolute Deviation (MAD) to detect anomalies.
- **🤖 Model Execution**: Compare responses from traditional models and LLMs like Genstruct and MixtraL.
- **📊 Evaluation**: 
  - **Traditional Model Evaluation**: Evaluate the traditional model responses to verify the accuracy of the root cause identification against standards specified in Target.json.
  - **LLM Evaluation**: Conduct a comprehensive evaluation, including a path-based evaluation using data from Graph.csv to assess the connections between the root cause and target nodes.


### 📁 Directory Structure

```
Root-Cause-Analysis-of-Cloud-Microservices/
├── 📄 Datametrics_Preprocessing.py     # Script for preprocessing metrics data
├── 📄 Groundtruthjson_Preprocessing.py # Script for preprocessing ground truth data
├── 📄 Path_Based_Evaluation.py         # Script for path-based evaluation
├── 📄 Pathtrace.py                     # Script for tracing paths in the graph
├── 📄 Runllm.py                        # Script to run large language models
└── 📄 requirements.txt                 # Required libraries and dependencies
```

## 🚀 Getting Started

### Prerequisites

Ensure you have Python installed on your system. This project is compatible with Python 3.8 and above.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/gayathritela/RCA-of-Cloud-Microservices.git
   cd RCA-of-Cloud-Microservices
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## 🛠 Usage

### Data Preparation

Use the preprocessing scripts to prepare your data:

```bash
python Datametrics_Preprocessing.py
python Groundtruthjson_Preprocessing.py
```

### Run Models

To execute the models and compare their performance:

```bash
python Runllm.py
```

### Evaluation

To evaluate the model predictions against the ground truth:

```bash
python Path_Based_Evaluation.py
```

## 👥 Contributing

Contributions are welcome! Here's how you can contribute:

- **Issues**: Report bugs or suggest features by creating an issue in the repository.
- **Pull Requests**: Fork the repository, make your changes, and submit a pull request for review.
- **Feedback**: Provide feedback on model usage and suggest improvements to enhance the project.

## 🙏 Acknowledgments

- **Supervisor**: Dr. Yan Liu
- **Researchers and Developers**: Gayathiri Elambooranan, Pranay Sood
- **Dataset**: Provided by the PetShop Dataset Authors

 ## 🤝 Connect


[Gayathiri Elambooranan](https://www.linkedin.com/in/gayathiri-elambooranan).

© 2024 Gayathiri Elambooranan.


