package main

import (
	"flag"
	"io"
	"log"
	"net"
)

func handle(src net.Conn, addr string) {
	defer src.Close()
	dest, err := net.Dial("tcp", addr)
	if err != nil {
		log.Fatalln("Unable to connect to server", err)
	}
	defer dest.Close()

	go func() {
		// Copy source request to destination
		if _, err := io.Copy(dest, src); err != nil {
			log.Fatalln(err)
		}
	}()

	// Copy destination response to source
	if _, err := io.Copy(src, dest); err != nil {
		log.Fatalln(err)
	}
}

func main() {
	var srvAddr, destAddr string
	flag.StringVar(&srvAddr, "addr", ":20080", "address the server listens to")
	flag.StringVar(&destAddr, "dest", ":20081", "proxy server address")
	flag.Parse()

	listener, err := net.Listen("tcp", srvAddr)
	if err != nil {
		log.Fatalln("Unable to bind port", err)
	}

	log.Println("Starting server on", srvAddr)
	for {
		conn, err := listener.Accept()
		log.Println("Connection received...")
		if err != nil {
			log.Println("Unable to accept connection", err)
		}
		go handle(conn, destAddr)
	}
}
