import pandas as pd
import boto3
import io
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

# SageMaker specific imports
from sagemaker.predictor import Predictor
from sagemaker.model import Model
import sagemaker
from sagemaker.processing import ScriptProcessor
from sagemaker.pytorch import PyTorchModel


# Load environment variables from .env file
load_dotenv()


# Retrieve AWS credentials securely from environment variables
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_default_region = os.getenv('AWS_DEFAULT_REGION')

# SageMaker session with authentication
sagemaker_session = sagemaker.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region=aws_default_region
)

# Define the data processing script
def data_processing_script(input_data, output_data):
    """
    This script processes data from S3, applies the Kalman filter, and stores the results.

    Args:
        input_data (str): Path to the input data file (S3 URI).
        output_data (str): Path to store the processed data (S3 URI).
    """

    s3_client = boto3.client("s3")

    # Download data from S3
    try:
        response = s3_client.get_object(Bucket=input_data.split('/')[2], Key=input_data.split('/')[-1])
        data = response["Body"].read()
    except Exception as e:
        print(f"Error downloading data from S3: {e}")
        return

    # Load data into DataFrame
    try:
        df = pd.read_csv(io.BytesIO(data))
    except Exception as e:
        print(f"Error loading data into DataFrame: {e}")
        return

    Y = df['Adj Close'].values
    S = Y.shape[0]

    # Define Kalman filter and smoother functions (removed unnecessary comments)
    def kalman_filter(param):
        Z = param[0]
        T = param[1]
        H = param[2]
        Q = param[3]

        # Initialize vectors
        u_predict, u_update, P_predict, P_update, v, F = {}, {}, {}, {}, {}, {}
        u_update[0] = Y[0]
        u_predict[0] = u_update[0]
        P_update[0] = np.var(Y) / 4
        P_predict[0] = T * P_update[0] * np.transpose(T) + Q

        Likelihood = 0
        for s in range(1, S):
            F[s] = Z * P_predict[s - 1] * np.transpose(Z) + H
            v[s] = Y[s - 1] - Z * u_predict[s - 1]
            u_update[s] = u_predict[s - 1] + P_predict[s - 1] * np.transpose(Z) * (1 / F[s]) * v[s]
            u_predict[s] = T * u_update[s]
            P_update[s] = P_predict[s - 1] - P_predict[s - 1] * np.transpose(Z) * (1 / F[s]) * Z * P_predict[s - 1]
            P_predict[s] = T * P_update[s] * np.transpose(T) + Q
            Likelihood += (1 / 2) * np.log(2 * np.pi) + (1 / 2) * np.log(abs(F[s])) + (1 / 2) * np.transpose(v[s]) * (1 / F[s]) * v[s]

        return Likelihood

    def kalman_smoother(params):
        Z = params[0]
        T = params[1]
        H = params[2]
        Q = params[3]

        # Initialize vectors (same as kalman_filter)
        u_predict, u_update, P_predict, P_update, v, F = {}, {}, {}, {}, {}, {}
        u_update[0] = Y[0]



# Retrieve AWS credentials securely from environment variables
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_default_region = os.getenv('AWS_DEFAULT_REGION')

# SageMaker session with authentication
sagemaker_session = sagemaker.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region=aws_default_region
)

# Define the data processing script
def data_processing_script(input_data, output_data):
    """
    This script processes data from S3, applies the Kalman filter, and stores the results.

    Args:
        input_data (str): Path to the input data file (S3 URI).
        output_data (str): Path to store the processed data (S3 URI).
    """

    s3_client = boto3.client("s3")

    # Download data from S3
    try:
        response = s3_client.get_object(Bucket=input_data.split('/')[2], Key=input_data.split('/')[-1])
        data = response["Body"].read()
    except Exception as e:
        print(f"Error downloading data from S3: {e}")
        return

    # Load data into DataFrame
    try:
        df = pd.read_csv(io.BytesIO(data))
    except Exception as e:
        print(f"Error loading data into DataFrame: {e}")
        return

    Y = df['Adj Close'].values
    S = Y.shape[0]

    def kalman_filter(param, *args):
        # Kalman filter implementation (same as before)

    def kalman_smoother(params, *args):
        # Kalman smoother implementation (same as before)

    param0 = np.array([0.85, 0.90, np.var(Y) / 45, np.var(Y) / 45])
    results = minimize(kalman_filter, param0, method='BFGS', options={'xtol': 1e-8, 'disp': True})

    param_star = results.x
    path = kalman_smoother(param_star, Y, S)
    Y_kalmanFilter = np.hstack(list(path.values()))
    Y_kalmanFilter = Y_kalmanFilter[::-1]

    timevec = np.linspace(1, S, S)
    RMSE = np.sqrt(np.mean((Y_kalmanFilter - Y) ** 2))

    # Store processed data (including RMSE) in CSV format
    processed_df = pd.DataFrame({'timevec': timevec,
                               'Actual Prices': Y,
                               'Kalman Filtered Prices': Y_kalmanFilter,
                               'RMSE': RMSE})
    processed_df.to_csv(output_data, index=False)

# Create the SageMaker processing script object
data_processor = ScriptProcessor(
    image_uri=sagemaker.image_uris.retrieve(framework="python", region=sagemaker_session.boto3.region_name, version="3.9"),
    script_mode="File",
    source_code="data_processing_script.py",  # Replace with your script filename
    role=sagemaker.get_execution_role(),
)

# Define the S3 URIs for input and output data
s3_bucket = "processed-meta-prices"  # Replace with your bucket name
input_data_uri = f"s3://{s3_bucket}/raw_data.csv"  # Replace with your input data path
output_data_uri = f"s3://{s3_bucket}/processed_data.csv"  # Replace with your output data path

# Process data using SageMaker
data_processor.run(
    inputs=[input_data_uri],
    outputs=[output_data_
