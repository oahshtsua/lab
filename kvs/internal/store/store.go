package store

import "errors"

var ErrKeyNotFound = errors.New("No such key")

type Store interface {
	Put(key string, value string) error
	Get(key string) (string, error)
	Delete(key string) error
}
