import json
import boto3
import os
import base64
import uuid

tableName = os.environ['DynamoDbName']

dynamodb = boto3.resource('dynamodb')

table = dynamodb.Table(tableName)


s3 = boto3.client('s3')
bucket = os.environ['BucketName']

headers = {
    "Access-Control-Allow-Headers": os.environ['headers'],
    "Access-Control-Allow-Methods": os.environ['methods'],
    "Access-Control-Allow-Credentials": bool(os.environ['credentials']),
    "Access-Control-Allow-Origin": os.environ['origin']

}

origin = os.environ['origin'].split(',')


def update(key, params):
    expression = 'set '

    values = {}

    for k, v in params.items():
        expression = expression + k + '=:' + k + ","
        values[':' + k] = v

    expression = expression[:-1]
    table.update_item(
        Key=key,
        UpdateExpression=expression,
        ExpressionAttributeValues=values,
        ReturnValues="UPDATED_NEW"
    )


def lambda_handler(event, context):
    print(event)

    httpMethod = event['httpMethod']
    userId = event['requestContext']['authorizer']['claims']['sub']
    imgId = event["queryStringParameters"]['imgId']

    if httpMethod == 'PUT':
        key = {
            'userId': userId,
            'imgId': imgId
        }

        body = json.loads(event["body"])
        update(key, body)

    if httpMethod == 'DELETE':
        file = 'cognito/ffs/' + event['requestContext']['authorizer']['claims']['sub'] + "/" + imgId
        s3.delete_object(Bucket=bucket, Key=file)


    response = {
        "isBase64Encoded": False,
        "statusCode": 204,
        "headers": headers,
    }

    print(response)
    return response
