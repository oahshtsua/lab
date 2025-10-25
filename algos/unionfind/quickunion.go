package main

type QuickUnionUF struct {
	ids   []int
	sizes []int
}

func (uf *QuickUnionUF) new(size int) {
	uf.ids = make([]int, size)
	uf.sizes = make([]int, size)
	for i := 0; i < size; i++ {
		uf.ids[i] = i
		uf.sizes[i] = 1
	}
}

func (uf *QuickUnionUF) root(i int) int {
	for i != uf.ids[i] {
		i = uf.ids[i]
	}
	return i
}

func (uf *QuickUnionUF) connected(p, q int) bool {
	return uf.root(p) == uf.root(q)
}

func (uf *QuickUnionUF) union(p, q int) {
	pRoot := uf.root(p)
	qRoot := uf.root(q)
	if pRoot == qRoot {
		return
	}
	if uf.sizes[pRoot] < uf.sizes[qRoot] {
		uf.ids[pRoot] = qRoot
		uf.sizes[qRoot] += uf.sizes[pRoot]
	} else {
		uf.ids[qRoot] = pRoot
		uf.sizes[pRoot] += uf.sizes[qRoot]
	}
}
