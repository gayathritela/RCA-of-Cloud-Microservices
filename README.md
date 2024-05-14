
# Root Cause Analysis of Cloud Microservices

## 🌐 Project Overview

![Project Overview](link-to-your-image.jpg)

This project focuses on analyzing root causes of anomalies in cloud microservices using both traditional models and Large Language Models (LLMs). The primary dataset used is the "PetShop Dataset," which is designed to facilitate root cause analysis in microservice environments.

### 🔑 Key Features

- **🔄 Data Preprocessing**: Transform and prepare data for analysis.
- **🚨 Anomaly Detection**: Use Median Absolute Deviation (MAD) to detect anomalies.
- **🤖 Model Execution**: Compare responses from traditional models and LLMs like Genstruct and MixtraL.
- **📊 Evaluation**: Assess model predictions against ground truth using path-based and other evaluation methods.

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

Contributions are welcome! Please feel free to submit pull requests to the repository.

## 🙏 Acknowledgments

- Thanks to Dr. Yan Liu for supervising this project.
- This project was inspired by the work described in "The PetShop Dataset — Finding Causes of Performance Issues across Microservices" by Hardt et al., 2023.
```

### Updated requirements.txt

```plaintext
# Fundamental packages
numpy>=1.18.5
pandas>=1.0.5
matplotlib>=3.2.2

# Machine learning libraries
scikit-learn>=0.23.1
gensim>=3.8.3

# Deep learning frameworks
torch>=1.5.0
transformers>=3.0.2
```

