import json
import os
import boto3
import json
from csv import reader

s3 = boto3.client('s3', config=boto3.session.Config(signature_version='s3v4', s3={'addressing_style': 'path'}))

bucketName = os.environ['BucketName']
bucket = boto3.resource('s3').Bucket(bucketName)

s3r = boto3.resource('s3')
s3r.Bucket(bucketName).download_file('outfits.csv', '/tmp/outfits.csv')

headers = {
    "Access-Control-Allow-Headers": os.environ['headers'],
    "Access-Control-Allow-Methods": os.environ['methods'],
    "Access-Control-Allow-Credentials": bool(os.environ['credentials']),
    "Access-Control-Allow-Origin": os.environ['origin']

}

origin = os.environ['origin'].split(',')


def lambda_handler(event, context):

    with open('/tmp/outfits.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        result = {}
        for row in csv_reader:
            datas = {
                "name": row[1],
                "clothes": row[2:]
            }

            result[row[0]] = datas

    response = {
        "isBase64Encoded": False,
        "statusCode": 200,
        "headers": headers,
        "body": json.dumps(result)
    }

    print(response)
    return response
