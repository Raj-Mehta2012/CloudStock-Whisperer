# KalmanFilter_Predictions_AWS
 
Create a Virtual Environment:
```python3 -m venv .awsvenv```

Activate Virtual Env:
```source .awsvenv/bin/activate```

Install libraries:
```pip3 install yfinance boto3 aws-shell```

Setup AWS Credentials:
```aws-shell```

Enter ```configure``` inside aws-shell and enter your AWS details

Create a s3 bucket called ```raw-meta-prices```
