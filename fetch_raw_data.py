import yfinance as yf
import boto3
from io import StringIO

ticker = 'META'
stock = yf.Ticker(ticker)
data = stock.history(period="5yr")
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
