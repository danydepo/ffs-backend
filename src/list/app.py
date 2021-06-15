import json
import os
import boto3

s3 = boto3.client('s3', config=boto3.session.Config(signature_version='s3v4', s3={'addressing_style': 'path'}))

bucketName = os.environ['BucketName']
bucket = boto3.resource('s3').Bucket(bucketName)

headers = {
    "Access-Control-Allow-Headers": os.environ['headers'],
    "Access-Control-Allow-Methods": os.environ['methods'],
    "Access-Control-Allow-Credentials": bool(os.environ['credentials']),
    "Access-Control-Allow-Origin": os.environ['origin']

}

origin = os.environ['origin'].split(',')


def lambda_handler(event, context):
    path = 'cognito/ffs/' + event['requestContext']['authorizer']['claims']['sub'] + "/"

    files = {}
    for obj in bucket.objects.filter(Prefix=path):
        files[os.path.basename(obj.key)] = s3.generate_presigned_url(
            ClientMethod='get_object',
            Params={'Bucket': bucketName, 'Key': obj.key}
        )

    response = {
        "isBase64Encoded": False,
        "statusCode": 200,
        "headers": headers,
        "body": json.dumps(files)
    }

    print(response)
    return response
