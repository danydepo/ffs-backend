import json
import boto3
import os
import base64
import uuid

tableName = os.environ['DynamoDbName']

dynamodb = boto3.resource('dynamodb')

table = dynamodb.Table(tableName)

headers = {
    "Access-Control-Allow-Headers": os.environ['headers'],
    "Access-Control-Allow-Methods": os.environ['methods'],
    "Access-Control-Allow-Credentials": bool(os.environ['credentials']),
    "Access-Control-Allow-Origin": os.environ['origin']

}

origin = os.environ['origin'].split(',')




def lambda_handler(event, context):
    userId = event['requestContext']['authorizer']['claims']['sub']
    imgId = event["queryStringParameters"]['imgId']

    response = table.get_item(Key={
        'userId': userId,
        'imgId': imgId
    })['Item']

    del response['userId']
    del response['imgId']


    response = {
        "isBase64Encoded": False,
        "statusCode": 200,
        "headers": headers,
        "body": json.dumps(response)
    }

    print(response)
    return response
