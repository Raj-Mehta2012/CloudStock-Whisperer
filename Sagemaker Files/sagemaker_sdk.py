import boto3
import os
from sagemaker import get_execution_role
from sagemaker.session import Session

aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_default_region = os.getenv('AWS_DEFAULT_REGION') 

# Configure the session and client
session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name= aws_default_region
)

sagemaker_client = session.client('sagemaker')

# Define the notebook instance parameters
notebook_instance_name = 'KalmanInstance'
instance_type = 'ml.t3.medium'
sagemaker_domain_id = 'd-e2i0gvmtl7pv'
role_arn = 'arn:aws:iam::149023223962:role/service-role/AmazonSageMaker-ExecutionRole-20240404T184210'  

# Wait for the notebook instance to be in service
waiter = sagemaker_client.get_waiter('notebook_instance_in_service')
waiter.wait(NotebookInstanceName=notebook_instance_name)
print(f"Notebook instance {notebook_instance_name} is now in service!")

# Describe the notebook instance to check its status
response = sagemaker_client.describe_notebook_instance(
    NotebookInstanceName=notebook_instance_name
)
print(response)

# Start the notebook instance
sagemaker_client.start_notebook_instance(
    NotebookInstanceName=notebook_instance_name
)

# Stop the notebook instance
# sagemaker_client.stop_notebook_instance(
#     NotebookInstanceName=notebook_instance_name
# )

# Using the SageMaker SDK to interact with the notebook instance
sagemaker_session = Session(boto_session=session)

file_path = 'Sagemaker Files/kalman-filter.ipynb'
sagemaker_session.upload_data(
    path=file_path,
    bucket='sagemaker-us-east-1-149023223962'
)

# Stop the notebook instance
sagemaker_client.stop_notebook_instance(
    NotebookInstanceName=notebook_instance_name
)
