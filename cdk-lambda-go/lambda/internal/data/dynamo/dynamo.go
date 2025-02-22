package dynamo

import (
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/dynamodb"
)

const TABLE_NAME = "cgl_users"

type DynamoDBClient struct {
	dataStore *dynamodb.DynamoDB
}

func NewDynamoDBClient() DynamoDBClient {
	session := session.Must(session.NewSession())
	db := dynamodb.New(session)
	return DynamoDBClient{dataStore: db}
}
