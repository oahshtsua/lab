package main

type QuickFindUF struct {
	ids []int
}

func (uf *QuickFindUF) new(size int) {
	uf.ids = make([]int, size)
	for i := 0; i < size; i++ {
		uf.ids[i] = i
	}
}

func (uf *QuickFindUF) connected(p, q int) bool {
	return uf.ids[p] == uf.ids[q]
}

func (uf *QuickFindUF) union(p, q int) {
	pId := uf.ids[p]
	qId := uf.ids[q]

	for i := 0; i < len(uf.ids); i++ {
		if uf.ids[i] == pId {
			uf.ids[i] = qId
		}
	}
}
