package logger

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

type FileTransactionLogger struct {
	events  chan<- Event
	errors  <-chan error
	lastSeq uint64
	file    *os.File
}

func (l *FileTransactionLogger) WriteDelete(key string) {
	l.events <- Event{Type: EventDelete, Key: key}
}

func (l *FileTransactionLogger) WritePut(key string, value string) {
	l.events <- Event{Type: EventPut, Key: key, Value: value}
}

func (l *FileTransactionLogger) Err() <-chan error {
	return l.errors
}

func (l *FileTransactionLogger) writeEvent(event Event) error {
	event.Sequence = l.lastSeq

	eventJson, err := json.Marshal(event)
	if err != nil {
		return err
	}

	_, err = l.file.Write(append(eventJson, '\n'))
	if err != nil {
		return err
	}
	l.file.Sync()
	return nil
}

func (l *FileTransactionLogger) ReadEvents() (<-chan Event, <-chan error) {
	scanner := bufio.NewScanner(l.file)
	eventChan := make(chan Event)
	errorChan := make(chan error)

	go func() {
		defer close(eventChan)
		defer close(errorChan)

		var event Event
		for scanner.Scan() {
			eventStr := scanner.Text()
			err := json.NewDecoder(strings.NewReader(eventStr)).Decode(&event)
			if err != nil {
				errorChan <- fmt.Errorf("Error parsing log event entry: %w", err)
				return
			}

			if l.lastSeq >= event.Sequence {
				errorChan <- fmt.Errorf("Transaction numbers out of order")
				return
			}

			l.lastSeq = event.Sequence
			eventChan <- event
		}
	}()
	return eventChan, errorChan
}

func (l *FileTransactionLogger) Run() {
	events := make(chan Event, 16)
	l.events = events

	errors := make(chan error, 1)
	l.errors = errors

	go func() {
		for event := range events {
			l.lastSeq++

			err := l.writeEvent(event)
			if err != nil {
				errors <- err
				return
			}

		}
	}()
}

func NewFileTransactionLogger(filename string) (*FileTransactionLogger, error) {
	file, err := os.OpenFile(filename, os.O_RDWR|os.O_APPEND|os.O_CREATE, 0644)
	if err != nil {
		return nil, fmt.Errorf("cannot open transaction log file: %w", err)
	}
	return &FileTransactionLogger{file: file}, nil
}
