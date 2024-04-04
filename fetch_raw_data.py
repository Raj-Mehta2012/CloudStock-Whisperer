import yfinance as yf
import boto3
from io import StringIO
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve environment variables
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_default_region = os.getenv('AWS_DEFAULT_REGION')

# Use the retrieved environment variables to configure boto3
session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_default_region
)

# Example: Create an S3 resource using the session
s3 = session.resource('s3')

# Now, you can use the S3 resource to interact with S3
# For example, list all buckets
for bucket in s3.buckets.all():
    print(bucket.name)

ticker = 'META'
data = yf.download('META', start='2022-03-30', end='2023-03-30')
csv_data = data.to_csv()

s3_resource = boto3.resource('s3')
bucket_name = 'raw-meta-prices'
object_name = f'{ticker}_stock_data.csv'

# Save the CSV to a string buffer
csv_buffer = StringIO()
csv_buffer.write(csv_data)
csv_buffer.seek(0)

# Upload the data to S3
s3_resource.Object(bucket_name, object_name).put(Body=csv_buffer.getvalue())
print(f'Successfully uploaded {object_name} to {bucket_name}')
