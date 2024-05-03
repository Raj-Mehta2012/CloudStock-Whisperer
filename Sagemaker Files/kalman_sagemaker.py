import pandas as pd
import boto3
import io
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from datetime import datetime

# SageMaker specific imports
from sagemaker.predictor import Predictor
from sagemaker.model import Model
import sagemaker
from sagemaker.processing import ScriptProcessor
from sagemaker.pytorch import PyTorchModel
from sagemaker import get_execution_role


# Load environment variables from .env file
load_dotenv()


# Retrieve AWS credentials securely from environment variables
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_default_region = os.getenv('AWS_DEFAULT_REGION')

# SageMaker session with authentication
sagemaker_session = sagemaker.Session()

s3_bucket = "processed-meta-prices"
sagemaker_output_bucket = "sagemaker-output"
s3_client = boto3.client("s3")

# List all objects in the bucket (assuming only one .csv file)
try:
  bucket_objects = s3_client.list_objects_v2(Bucket=s3_bucket)["Contents"]
except Exception as e:
  print(f"Error listing objects in S3 bucket: {e}")

if not bucket_objects:
  print("No objects found in the bucket.")
  exit()  # Exit the script if no objects exist

# Assuming the first object is the CSV file
file_key = bucket_objects[0]["Key"]

# Download data from S3
try:
  response = s3_client.get_object(Bucket=s3_bucket, Key=file_key)
  data = response["Body"].read()
except Exception as e:
  print(f"Error downloading data from S3 for {file_key}: {e}")
  exit() 

#Load data into Dataframe
try:
  df = pd.read_csv(io.BytesIO(data)) 
  print(f"Loaded data from: {file_key}")
except Exception as e:
  print(f"Error loading data into DataFrame for {file_key}: {e}")
  exit()

sagemaker_session = sagemaker.Session()
role = get_execution_role(sagemaker_session)

script_path = 'kalman_filter_script.py'  # Replace with the name of your script
estimator = sagemaker.estimator.Estimator(
    entry_point=script_path,
    role=role,
    instance_count=1,
    instance_type='ml.m5.medium',
    output_path=f's3://{sagemaker_output_bucket}',
    sagemaker_session=sagemaker_session
)

# Launch the training job
estimator.fit({'train': df})

# Deploy the model
predictor = estimator.deploy(initial_instance_count=1, instance_type='ml.m5.medium')
