package main

import (
	"cdk-go-lambda/internal/data"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/aws/aws-lambda-go/events"
)

func (app *Application) registerUserHandler(req events.APIGatewayProxyRequest) (events.APIGatewayProxyResponse, error) {
	var registerRequest struct {
		Username string `json:"username"`
		Password string `json:"password"`
	}
	err := json.Unmarshal([]byte(req.Body), &registerRequest)
	if err != nil {
		return events.APIGatewayProxyResponse{
			Body:       "Invalid request.",
			StatusCode: http.StatusBadRequest,
		}, err
	}

	if registerRequest.Username == "" || registerRequest.Password == "" {
		return events.APIGatewayProxyResponse{
			Body:       "Invalid request.",
			StatusCode: http.StatusBadRequest,
		}, err
	}

	existingUser, err := app.db.GetUser(registerRequest.Username)
	if err != nil {
		return events.APIGatewayProxyResponse{
			Body:       "Internal server error",
			StatusCode: http.StatusInternalServerError,
		}, err
	}

	if existingUser != nil {
		return events.APIGatewayProxyResponse{
			Body:       "User already exists",
			StatusCode: http.StatusConflict,
		}, err
	}

	hashedPw, err := getPasswordHash(registerRequest.Password)
	if err != nil {
		return events.APIGatewayProxyResponse{
			Body:       "Internal server error",
			StatusCode: http.StatusInternalServerError,
		}, err
	}

	user := data.User{
		Username: registerRequest.Username,
		Password: string(hashedPw),
	}

	err = app.db.CreateUser(&user)
	if err != nil {
		return events.APIGatewayProxyResponse{
			Body:       "Internal server error",
			StatusCode: http.StatusInternalServerError,
		}, err
	}

	return events.APIGatewayProxyResponse{
		Body:       "User creation successful.",
		StatusCode: http.StatusCreated,
	}, nil
}

func (app *Application) loginUserHandler(req events.APIGatewayProxyRequest) (events.APIGatewayProxyResponse, error) {
	var loginRequest struct {
		Username string `json:"username"`
		Password string `json:"password"`
	}
	err := json.Unmarshal([]byte(req.Body), &loginRequest)
	if err != nil {
		return events.APIGatewayProxyResponse{
			Body:       "Invalid request.",
			StatusCode: http.StatusBadRequest,
		}, err
	}

	user, err := app.db.GetUser(loginRequest.Username)
	if err != nil {
		return events.APIGatewayProxyResponse{
			Body:       "Internal server error",
			StatusCode: http.StatusInternalServerError,
		}, err
	}

	if user == nil {
		return events.APIGatewayProxyResponse{
			Body:       "Invalid login credentials",
			StatusCode: http.StatusBadRequest,
		}, err
	}

	ok, err := validateUserPassword(user.Password, loginRequest.Password)
	if err != nil {
		return events.APIGatewayProxyResponse{
			Body:       "Internal server error",
			StatusCode: http.StatusInternalServerError,
		}, err
	}
	if !ok {
		return events.APIGatewayProxyResponse{
			Body:       "Invalid user credentials",
			StatusCode: http.StatusBadRequest,
		}, nil
	}

	token, err := createJWT(*user)
	if err != nil {
		return events.APIGatewayProxyResponse{
			Body:       "Internal server error",
			StatusCode: http.StatusInternalServerError,
		}, err
	}

	return events.APIGatewayProxyResponse{
		Body:       fmt.Sprintf("Token: %s", token),
		StatusCode: http.StatusOK,
	}, nil
}

func (app *Application) someProtectedHandler(req events.APIGatewayProxyRequest) (events.APIGatewayProxyResponse, error) {
	return events.APIGatewayProxyResponse{
		Body:       "Protected route response",
		StatusCode: http.StatusOK,
	}, nil
}
