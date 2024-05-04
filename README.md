# Kalman Filter and XGBoost Predictions on AWS

This project aims to implement a Kalman filter and XGBoost regression model for stock price predictions on the AWS cloud platform. It utilizes various AWS services, including S3, Glue, and SageMaker, to orchestrate the data processing, model training, and deployment pipeline. The project leverages a serverless architecture, ensuring scalability and cost-effectiveness.

## Prerequisites

Before proceeding, ensure that you have the following prerequisites:

- Python 3.x installed
- AWS account with appropriate permissions (IAM roles, policies, and access keys)
- AWS CLI installed and configured with valid credentials

## Project Setup

### 1. Virtual Environment Creation

Create a virtual environment to isolate the project dependencies using Python's built-in `venv` module:

```bash
python3 -m venv .awsvenv
```

Activate the virtual environment:

```bash
source .awsvenv/bin/activate  # on Windows use `.awsvenv\Scripts\activate`
```
Install the required Python packages specified in the ```requirements.txt``` file:

```bash
pip install -r requirements.txt
```





