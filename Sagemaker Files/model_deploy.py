import sagemaker
from sagemaker import get_execution_role
import boto3
import pandas as pd
import io
import numpy as np
from scipy.optimize import minimize
from datetime import datetime
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

load_dotenv()

# Set up AWS credentials and region
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION')

# Create a SageMaker session
sagemaker_session = sagemaker.Session()

# Get the execution role for SageMaker
role = get_execution_role(sagemaker_session)

# Create an S3 client
s3_client = boto3.client('s3')

# Specify the S3 bucket
s3_bucket = "processed-meta-prices"
sagemaker_output_bucket = "sagemaker-output"

try:
    bucket_objects = s3_client.list_objects_v2(Bucket=s3_bucket)["Contents"]
except Exception as e:
    print(f"Error listing objects in S3 bucket: {e}")

if not bucket_objects:
    print("No objects found in the bucket.")
    exit()

# Assuming the first object is the CSV file
file_key = bucket_objects[0]["Key"]

try:
    response = s3_client.get_object(Bucket=s3_bucket, Key=file_key)
    data = response["Body"].read()
except Exception as e:
    print(f"Error downloading data from S3 for {file_key}: {e}")
    exit()

try:
    df = pd.read_csv(io.BytesIO(data))
    print(f"Loaded data from: {file_key}")
except Exception as e:
    print(f"Error loading data into DataFrame for {file_key}: {e}")
    exit()

# Your model functions
def kalman_filter(param,*args):
    # initialize params
    Z = param[0]
    T = param[1]
    H = param[2]
    Q = param[3]
    # initialize vector values:
    u_predict,  u_update,  P_predict, P_update, v, F = {},{},{},{},{},{}
    u_update[0] = Y[0]
    u_predict[0] = u_update[0]
    P_update[0] = np.var(Y)/4
    P_predict[0] =  T*P_update[0]*np.transpose(T)+Q
    Likelihood = 0
    for s in range(1, S):
        F[s] = Z*P_predict[s-1]*np.transpose(Z)+H
        v[s]= Y[s-1]-Z*u_predict[s-1]
        u_update[s] = u_predict[s-1]+P_predict[s-1]*np.transpose(Z)*(1/F[s])*v[s]
        u_predict[s] = T*u_update[s]
        P_update[s] = P_predict[s-1]-P_predict[s-1]*np.transpose(Z)*(1/F[s])*Z*P_predict[s-1]
        P_predict[s] = T*P_update[s]*np.transpose(T)+Q
        Likelihood += (1/2)*np.log(2*np.pi)+(1/2)*np.log(abs(F[s]))+(1/2)*np.transpose(v[s])*(1/F[s])*v[s]

    return Likelihood


def kalman_smoother(params, *args):
    # initialize params
    Z = params[0]
    T = params[1]
    H = params[2]
    Q = params[3]
    # initialize vector values:
    u_predict,  u_update,  P_predict, P_update, v, F = {},{},{},{},{},{}
    u_update[0] = Y[0]
    u_predict[0] = u_update[0]
    P_update[0] = np.var(Y)/4
    P_predict[0] =  T*P_update[0]*np.transpose(T)+Q
    for s in range(1, S):
        F[s] = Z*P_predict[s-1]*np.transpose(Z)+H
        v[s]=Y[s-1]-Z*u_predict[s-1]
        u_update[s] = u_predict[s-1]+P_predict[s-1]*np.transpose(Z)*(1/F[s])*v[s]
        u_predict[s] = T*u_update[s]
        P_update[s] = P_predict[s-1]-P_predict[s-1]*np.transpose(Z)*(1/F[s])*Z*P_predict[s-1]
        P_predict[s] = T*P_update[s]*np.transpose(T)+Q

    u_smooth, P_smooth = {}, {}
    u_smooth[S-1] = u_update[S-1]
    P_smooth[S-1] = P_update[S-1]
    for t in range(S-1, 0, -1):
        u_smooth[t-1] = u_update[t] + P_update[t]*np.transpose(T)/P_predict[t]*(u_smooth[t]-T*u_update[s])
        P_smooth[t-1] = P_update[t] + P_update[t]*np.transpose(T)/P_predict[t]*(P_smooth[t]-P_predict[t])/P_predict[t]*T*P_update[t]

    # del u_update[-1]
    smooth_path = u_smooth
    return smooth_path

# Create an input data channel for SageMaker
input_data = sagemaker_session.upload_data(
    path=f's3://{s3_bucket}/{file_key}',
    key_prefix='data',
    bucket=s3_bucket
)

# Define the estimator and set the entry point script
script_path = 'kalman_filter_script.py'  # Replace with the name of your script
estimator = sagemaker.estimator.Estimator(
    entry_point=script_path,
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    output_path=f's3://{sagemaker_output_bucket}',
    sagemaker_session=sagemaker_session
)

# Launch the training job
estimator.fit({'train': input_data})

# Deploy the model
predictor = estimator.deploy(initial_instance_count=1, instance_type='ml.m5.medium')

