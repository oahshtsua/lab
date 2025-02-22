.PHONY: build
build:
	GOOS=linux GOARCH=amd64 go build -o bootstrap ./cmd/api/
	zip function.zip bootstrap

.PHONY: clean
clean:
	rm -rf function.zip bootstrap
