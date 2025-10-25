package main

import "testing"

func TestQuickFindUF(t *testing.T) {
	var uf QuickFindUF
	uf.new(10)

	uf.union(4, 3)
	uf.union(3, 8)
	uf.union(6, 5)
	uf.union(9, 4)
	uf.union(2, 1)

	testCases := []struct {
		p, q     int
		expected bool
	}{
		{0, 0, true},
		{4, 3, true},
		{3, 4, true},
		{8, 9, true},
		{0, 7, false},
		{3, 1, false},
	}
	for _, tc := range testCases {
		result := uf.connected(tc.p, tc.q)
		if result != tc.expected {
			t.Errorf("connected(%d, %d) = %t; expected = %t", tc.p, tc.q, result, tc.expected)
		}
	}
}

func TestQuickUnionUF(t *testing.T) {
	var uf QuickUnionUF
	uf.new(10)

	uf.union(4, 3)
	uf.union(3, 8)
	uf.union(6, 5)
	uf.union(9, 4)
	uf.union(2, 1)

	testCases := []struct {
		p, q     int
		expected bool
	}{
		{0, 0, true},
		{4, 3, true},
		{3, 4, true},
		{8, 9, true},
		{0, 7, false},
		{3, 1, false},
	}
	for _, tc := range testCases {
		result := uf.connected(tc.p, tc.q)
		if result != tc.expected {
			t.Errorf("connected(%d, %d) = %t; expected = %t", tc.p, tc.q, result, tc.expected)
		}
	}
}
