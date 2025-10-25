package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"net"
)

func echo(conn net.Conn) {
	defer conn.Close()
	_, err := io.Copy(conn, conn)
	if err != nil {
		log.Fatalln("Unable to read/write data: ", err)
	}
}

func main() {
	var port int
	flag.IntVar(&port, "port", 20081, "Network port to listen on")
	flag.Parse()

	listener, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		log.Fatalln("Unable to bind port")
	}

	log.Println("Starting server on port", port)
	for {
		conn, err := listener.Accept()
		log.Println("Connection received...")
		if err != nil {
			log.Println("Unable to accept connection", err)
		}
		go echo(conn)
	}
}
