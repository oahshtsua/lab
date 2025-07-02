package store

import "sync"

type MemoryStore struct {
	sync.RWMutex
	inner map[string]string
}

func NewMemoryStore() *MemoryStore {
	return &MemoryStore{
		inner: make(map[string]string),
	}
}

func (s *MemoryStore) Put(key string, value string) error {
	s.Lock()
	s.inner[key] = value
	s.Unlock()
	return nil
}

func (s *MemoryStore) Get(key string) (string, error) {
	s.RLock()
	val, ok := s.inner[key]
	s.RUnlock()
	if !ok {
		return "", ErrKeyNotFound
	}
	return val, nil
}

func (s *MemoryStore) Delete(key string) error {
	s.Lock()
	delete(s.inner, key)
	s.Unlock()
	return nil
}
