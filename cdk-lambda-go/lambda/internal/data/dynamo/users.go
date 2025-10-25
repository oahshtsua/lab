package dynamo

import (
	"cdk-go-lambda/internal/data"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/dynamodb"
	"github.com/aws/aws-sdk-go/service/dynamodb/dynamodbattribute"
)

func (c DynamoDBClient) GetUser(username string) (*data.User, error) {
	var user data.User
	res, err := c.dataStore.GetItem(&dynamodb.GetItemInput{
		TableName: aws.String(TABLE_NAME),
		Key: map[string]*dynamodb.AttributeValue{
			"username": {
				S: aws.String(username),
			},
		},
	})

	if err != nil {
		return nil, err
	}

	if res.Item == nil {
		return nil, nil
	}

	err = dynamodbattribute.UnmarshalMap(res.Item, &user)
	if err != nil {
		return nil, err
	}
	return &user, nil
}

func (c DynamoDBClient) CreateUser(user *data.User) error {
	item := &dynamodb.PutItemInput{
		TableName: aws.String(TABLE_NAME),
		Item: map[string]*dynamodb.AttributeValue{
			"username": {
				S: aws.String(user.Username),
			},
			"password": {
				S: aws.String(user.Password),
			},
		},
	}
	_, err := c.dataStore.PutItem(item)
	if err != nil {
		return err
	}
	return nil

}
