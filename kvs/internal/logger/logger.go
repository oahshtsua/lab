package logger

type EventType byte

const (
	_ EventType = iota
	EventDelete
	EventPut
)

type Event struct {
	Sequence uint64    `json:"seq"`
	Type     EventType `json:"type"`
	Key      string    `json:"key"`
	Value    string    `json:"value"`
}

type TransactionLogger interface {
	WriteDelete(key string)
	WritePut(key string, value string)
	Err() <-chan error
	Run()
	ReadEvents() (<-chan Event, <-chan error)
}
