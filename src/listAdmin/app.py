import json
import os
import boto3
import json
from csv import reader

s3 = boto3.client('s3', config=boto3.session.Config(signature_version='s3v4', s3={'addressing_style': 'path'}))

bucketName = os.environ['BucketName']
bucketNameGT = os.environ['BucketNameGT']
bucket = boto3.resource('s3').Bucket(bucketName)

s3r = boto3.resource('s3')
s3r.Bucket(bucketNameGT).download_file('clothesGT.csv', '/tmp/clothesGT.csv')

headers = {
    "Access-Control-Allow-Headers": os.environ['headers'],
    "Access-Control-Allow-Methods": os.environ['methods'],
    "Access-Control-Allow-Credentials": bool(os.environ['credentials']),
    "Access-Control-Allow-Origin": os.environ['origin']

}

origin = os.environ['origin'].split(',')


def lambda_handler(event, context):
    s3r.Bucket(bucketNameGT).download_file('clothesGT.csv', '/tmp/clothesGT.csv')
    discovered = []
    clothesGT = []
    header = True
    with open('/tmp/clothesGT.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            if header:
                header = False
                continue
            if row[0] + row[3] not in discovered:
                discovered.append(row[0] + row[3])
                colors = {}
                for c in row[2].split(','):
                    tmp = c.split('(')
                    color = tmp[0]
                    percentage = tmp[1][:-1]
                    colors[color] = percentage

                clothesGT.append(
                    {
                        "category": row[0],
                        "name": row[1],
                        "mainColor": row[3],
                        "colors": colors
                    }
                )

    for c in clothesGT:
        c['url'] = s3.generate_presigned_url(
            ClientMethod='get_object',
            Params={'Bucket': bucketNameGT, 'Key': 'images2/' + c['category'] + '/' + c['name']}
        )

    response = {
        "isBase64Encoded": False,
        "statusCode": 200,
        "headers": headers,
        "body": json.dumps(clothesGT)
    }

    print(response)
    return response
