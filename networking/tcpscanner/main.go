// tcp port scanner

package main

import (
	"flag"
	"fmt"
	"net"
	"slices"
)

const (
	numWorkers = 100
	startPort  = 1
	endPort    = 1024
)

func worker(host string, ports, results chan int) {
	for port := range ports {
		addr := fmt.Sprintf("%s:%d", host, port)
		conn, err := net.Dial("tcp", addr)
		if err != nil {
			results <- 0
			continue
		}
		conn.Close()
		results <- port
	}
}

func main() {
	var host string
	flag.StringVar(&host, "host", "localhost", "the host to scan for open ports")
	flag.Parse()
	ports := make(chan int, numWorkers)
	results := make(chan int)

	for i := 0; i < int(numWorkers); i++ {
		go worker(host, ports, results)
	}

	go func() {
		for i := startPort; i <= endPort; i++ {
			ports <- i
		}
	}()

	var openPorts []int
	for i := startPort; i <= endPort; i++ {
		port := <-results
		if port != 0 {
			openPorts = append(openPorts, port)
		}
	}
	close(ports)
	close(results)

	slices.Sort(openPorts)
	for _, port := range openPorts {
		fmt.Printf("Port %d open\n", port)
	}
}
