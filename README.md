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

Additionally, install the following libraries for interacting with AWS services and retrieving stock data (Only do this if you are not using virtual environment):

```bash
pip3 install yfinance boto3 aws-shell
```

### 2. AWS Credentials Configuration

Set up your AWS credentials by running the `aws-shell` command, which provides a user-friendly interface for configuring the AWS CLI:

```bash
aws-shell
```

Within the AWS Shell, enter `configure` and provide your AWS access key, secret key, and default region.

### 3. Environment Variables

Rename the `sample-env` file to `.env` and update the environment variables with your AWS account credentials, including the access key, secret key, and default region.

### 4. S3 Bucket Creation

Sign in to the AWS Console, navigate to the S3 service, and create two buckets:

- `raw-meta-prices`: This bucket will store the raw stock price data retrieved from Yahoo Finance.
- `processed-meta-prices`: This bucket will store the preprocessed data after the Glue job has transformed it.

Ensure that the buckets are created in the same region as specified in your AWS configuration.

### 5. Data Ingestion

With the virtual environment activated, run the Python script `ingest_data.py` (or a similar script) to download stock prices from Yahoo Finance using the `yfinance` library. The script should then upload the raw data to the `raw-meta-prices` S3 bucket.

### 6. Glue Job and Crawler

#### Glue Crawler

1. In the AWS Glue service, create a new crawler to catalog the data stored in the `raw-meta-prices` S3 bucket.
2. Configure the crawler to periodically scan the bucket for new data and update the AWS Glue Data Catalog accordingly.

#### Glue Job

1. Create a new Glue job to preprocess the raw stock price data.
2. Add the PySpark code from the `Glue Files` directory in the repository to the Glue job script.
3. Configure the job to read data from the `raw-meta-prices` bucket and write the processed data as CSV files to the `processed-meta-prices` bucket.
4. Set up triggers or schedules to run the Glue job automatically as new data arrives in the `raw-meta-prices` bucket.

The Glue job should handle tasks such as data cleaning, feature engineering, and any necessary transformations required for the downstream model training process.

### 7. SageMaker Notebook Instance

1. Navigate to the SageMaker Console and create a new domain (optional) and a notebook instance. For this project, we recommend using the `ml.t2.medium` instance type, which provides a balance between compute power and cost-effectiveness.
2. Once the notebook instance is created, access the JupyterLab environment and upload the `.ipynb` file from the `Sagemaker Files` directory in the repository.
3. In the notebook, configure the necessary AWS credentials and specify the S3 bucket locations for input data and model artifacts.
4. Run the notebook file, which will load the preprocessed data from the `processed-meta-prices` bucket, train the XGBoost model, and save the model artifacts and output to a new S3 bucket.

### 8. Model Deployment

After the model is trained, you can deploy it as a SageMaker endpoint using the following code:

```python
from sagemaker.serializers import CSVSerializer

xgb_predictor = xgb.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium',
    serializer=CSVSerializer()
)
```

The deployed model can then be used for making predictions on new data, either on the local system or within the AWS ecosystem. You can invoke the model endpoint using the SageMaker SDK or integrate it with other AWS services like Lambda or API Gateway for real-time inference.

For detailed step-by-step instructions on deploying and using the model, refer to the AWS documentation: [Deploying a Model in Amazon SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/ex1-model-deployment.html#ex1-deploy-model-sdk-use-endpoint).

## License

This project is licensed under the [MIT License](LICENSE).
```
