package main

import (
	"cdk-go-lambda/internal/data"
	"cdk-go-lambda/internal/data/dynamo"
	"net/http"

	"github.com/aws/aws-lambda-go/events"
	"github.com/aws/aws-lambda-go/lambda"
)

type Application struct {
	db data.DataStore
}

func main() {
	dynamodb := dynamo.NewDynamoDBClient()
	app := Application{
		db: dynamodb,
	}
	lambda.Start(func(req events.APIGatewayProxyRequest) (events.APIGatewayProxyResponse, error) {
		switch req.Path {
		case "/register":
			return app.registerUserHandler(req)
		case "/login":
			return app.loginUserHandler(req)
		case "/protected":
			return ValidateJWT(app.someProtectedHandler)(req)
		default:
			return events.APIGatewayProxyResponse{Body: "Not found.", StatusCode: http.StatusNotFound}, nil
		}
	})
}
