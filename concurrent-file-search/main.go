package main

import (
	"flag"
	"log"
	"os"
	"path"
	"sync"
)

var (
	matches []string
	wg      sync.WaitGroup
	lock    sync.Mutex
)

func main() {

	var dir, target string
	flag.StringVar(&dir, "d", ".", "the directory to search in")
	flag.StringVar(&target, "t", "", "the target file to search for")
	flag.Parse()
	if target == "" {
		return
	}

	wg.Add(1)
	go searchFile(dir, target)
	wg.Wait()

	for _, file := range matches {
		log.Println("Matched: \n", file)
	}

}

func searchFile(root string, target string) {
	defer wg.Done()
	log.Println("Searching: ", root)
	dirContents, err := os.ReadDir(root)
	if err != nil {
		log.Fatal(err)
	}

	for _, item := range dirContents {
		if item.IsDir() {
			wg.Add(1)
			go searchFile(path.Join(root, item.Name()), target)
		} else if item.Name() == target {
			lock.Lock()
			matches = append(matches, path.Join(root, item.Name()))
			lock.Unlock()
		}
	}
}
