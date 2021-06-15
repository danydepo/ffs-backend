import json
import boto3
import os
import base64
import uuid

s3 = boto3.client('s3')
bucket = os.environ['BucketName']

headers = {
    "Access-Control-Allow-Headers": os.environ['headers'],
    "Access-Control-Allow-Methods": os.environ['methods'],
    "Access-Control-Allow-Credentials": bool(os.environ['credentials']),
    "Access-Control-Allow-Origin": os.environ['origin']

}

origin = os.environ['origin'].split(',')


def lambda_handler(event, context):
    request = json.loads(event["body"])

    image = base64.b64decode(request["file"])

    path = 'cognito/ffs/' + event['requestContext']['authorizer']['claims']['sub'] + "/"
    filename = str(uuid.uuid4().hex) + request["format"]

    s3.put_object(Bucket=bucket, Key=path + filename, Body=image)

    response = {
        "isBase64Encoded": False,
        "statusCode": 200,
        "headers": headers,
        "body": json.dumps({"id": filename})
    }

    print(response)
    return response
