package main

import (
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"

	"github.com/oahshtsua/lab/kvs/internal/logger"
	"github.com/oahshtsua/lab/kvs/internal/store"
)

type application struct {
	port    int
	tlogger logger.TransactionLogger
	store   store.Store
}

func main() {
	store := store.NewMemoryStore()
	log.Println("Initializing transaction log...")
	tlogger, err := initializeTransactionLog(store, "transactions.jsonl")
	if err != nil {
		log.Fatalln(err)
	}

	app := application{
		port:    5000,
		tlogger: tlogger,
		store:   store,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/key/{key}", app.getKeyHandler)
	mux.HandleFunc("PUT /v1/key/{key}", app.putKeyHandler)
	mux.HandleFunc("DELETE /v1/key/{key}", app.deleteKeyHandler)

	srv := &http.Server{
		Addr:    fmt.Sprintf(":%d", app.port),
		Handler: mux,
	}

	log.Printf("Starting server on port: %d", app.port)
	log.Fatalln(srv.ListenAndServe())
}

func (app *application) getKeyHandler(w http.ResponseWriter, r *http.Request) {
	key := r.PathValue("key")
	val, err := app.store.Get(key)
	if err != nil {
		if errors.Is(err, store.ErrKeyNotFound) {
			w.WriteHeader(http.StatusNotFound)
		} else {
			w.WriteHeader(http.StatusInternalServerError)
		}
		return
	}
	w.Write([]byte(val))
}

func (app *application) putKeyHandler(w http.ResponseWriter, r *http.Request) {
	key := r.PathValue("key")

	value, err := io.ReadAll(r.Body)
	defer r.Body.Close()

	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		return
	}

	err = app.store.Put(key, string(value))
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		return
	}
	app.tlogger.WritePut(key, string(value))
	w.WriteHeader(http.StatusCreated)
}

func (app *application) deleteKeyHandler(w http.ResponseWriter, r *http.Request) {
	key := r.PathValue("key")
	err := app.store.Delete(key)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		return
	}
	app.tlogger.WriteDelete(key)
	w.WriteHeader(http.StatusNoContent)
}

func initializeTransactionLog(store store.Store, filename string) (*logger.FileTransactionLogger, error) {
	var err error

	tlog, err := logger.NewFileTransactionLogger(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to create transaction log: %w", err)
	}

	events, errors := tlog.ReadEvents()
	event, ok := logger.Event{}, true

	for ok && err == nil {
		select {
		case err, ok = <-errors:
		case event, ok = <-events:
			switch event.Type {
			case logger.EventDelete:
				err = store.Delete(event.Key)
			case logger.EventPut:
				err = store.Put(event.Key, event.Value)
			}
		}
	}
	tlog.Run()

	return tlog, err
}
