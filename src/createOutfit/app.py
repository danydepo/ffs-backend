import json
import boto3
import os
from botocore.exceptions import ClientError
import uuid

s3r = boto3.resource('s3')

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
    print(event)
    try:
        s3r.Bucket(bucket).download_file('outfits.csv', '/tmp/outfits.csv')

    except ClientError as e:
        print("new")

    body = json.loads(event['body'])
    print(body)
    line = str(uuid.uuid4().hex) + ',' + body['name']

    for o in body['clothes']:
        line = line + ',' + o

    with open('/tmp/outfits.csv', 'a') as fd:
        fd.write(line + '\n')

    fd.close()

    s3.upload_file('/tmp/outfits.csv', bucket, 'outfits.csv')

    response = {
        "isBase64Encoded": False,
        "statusCode": 204,
        "headers": headers,
    }

    print(response)
    return response