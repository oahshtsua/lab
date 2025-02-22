package main

import (
	"net/http"
	"time"

	"github.com/aws/aws-lambda-go/events"
)

type APIGatewayHandler func(request events.APIGatewayProxyRequest) (events.APIGatewayProxyResponse, error)

func ValidateJWT(next APIGatewayHandler) APIGatewayHandler {
	return func(request events.APIGatewayProxyRequest) (events.APIGatewayProxyResponse, error) {

		token := extractToken(request.Headers)
		if token == "" {
			return events.APIGatewayProxyResponse{
				Body:       "Missing auth token",
				StatusCode: http.StatusUnauthorized,
			}, nil
		}

		claims, err := parseJWT(token)
		if err != nil {
			return events.APIGatewayProxyResponse{
				Body:       "Unauthorized",
				StatusCode: http.StatusUnauthorized,
			}, nil

		}

		expires := int64(claims["expires"].(float64))
		if expires < time.Now().Unix() {
			return events.APIGatewayProxyResponse{
				Body:       "Token expired",
				StatusCode: http.StatusUnauthorized,
			}, nil
		}
		return next(request)
	}
}
