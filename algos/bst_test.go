package main

import "testing"

func equalSlices(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}

	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestTraverseInorder(t *testing.T) {
	r := &TreeNode{
		value: 32,
		left: &TreeNode{
			value: 21,
			left: &TreeNode{
				value: 7,
			},
			right: &TreeNode{
				value: 28,
			},
		},
		right: &TreeNode{
			value: 38,
			left: &TreeNode{
				value: 35,
			},
			right: &TreeNode{
				value: 47,
			},
		},
	}
	bst := BinarySearchTree{root: r}
	expected := []int{7, 21, 28, 32, 35, 38, 47}
	result := bst.TraverseInorder()
	if !equalSlices(expected, result) {
		t.Fatalf("Expected: %v\nGot: %v\n", expected, result)
	}
}

func TestAdd(t *testing.T) {
	bst := &BinarySearchTree{}
	nodes := []int{32, 21, 38, 47, 28, 7, 35}

	for _, node := range nodes {
		bst.Add(node)
	}

	if bst.root.value != nodes[0] {
		t.Fatalf("Expected: %d\nGot: %d\n", nodes[0], bst.root.value)
	}

	expected := []int{7, 21, 28, 32, 35, 38, 47}
	result := bst.TraverseInorder()
	if !equalSlices(expected, result) {
		t.Fatalf("Expected: %v\nGot: %v\n", expected, result)
	}
}

func TestSearch(t *testing.T) {
	bst := &BinarySearchTree{}
	nodes := []int{32, 21, 38, 47, 28, 7, 35}

	for _, node := range nodes {
		bst.Add(node)
	}

	t.Run("searching absent value", func(t *testing.T) {
		result := bst.Search(99)

		if result != nil {
			t.Fatalf("Expected: %v\nGot: %v\n", nil, result)
		}
	})

	t.Run("searching present value", func(t *testing.T) {
		expected := bst.root.right.left
		result := bst.Search(35)

		if result != expected {
			t.Fatalf("Expected: %v\nGot: %v\n", expected, result)
		}
	})
}

func TestRemove(t *testing.T) {
	bst := &BinarySearchTree{}
	nodes := []int{32, 21, 38, 47, 28, 7, 35}

	for _, node := range nodes {
		bst.Add(node)
	}

	t.Run("removing leaf node", func(t *testing.T) {
		bst.Remove(bst.Search(28))
		expected := []int{7, 21, 32, 35, 38, 47}
		result := bst.TraverseInorder()
		if !equalSlices(expected, result) {
			t.Fatalf("Expected: %v\nGot: %v\n", expected, result)
		}
	})

	t.Run("removing root node", func(t *testing.T) {
		bst.Remove(bst.Search(32))
		expectedRoot := bst.Search(35)
		resultRoot := bst.root
		if resultRoot != expectedRoot {
			t.Fatalf("Expected: %v\nGot: %v\n", expectedRoot, resultRoot)
		}

		expected := []int{7, 21, 35, 38, 47}
		result := bst.TraverseInorder()
		if !equalSlices(expected, result) {
			t.Fatalf("Expected: %v\nGot: %v\n", expected, result)
		}
	})

	t.Run("removing internal node", func(t *testing.T) {
		bst.Remove(bst.Search(21))
		expected := []int{7, 35, 38, 47}
		result := bst.TraverseInorder()
		if !equalSlices(expected, result) {
			t.Fatalf("Expected: %v\nGot: %v\n", expected, result)
		}
	})
}

func TestBulkConstruct(t *testing.T) {
	arr := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	bst := BulkConstruct(arr)

	result := bst.TraverseInorder()
	if !equalSlices(arr, result) {
		t.Fatalf("Expected: %v\nGot: %v\n", arr, result)
	}

	t.Run("checking root", func(t *testing.T) {
		mid := (0 + len(arr) - 1) / 2
		if bst.root.value != arr[mid] {
			t.Fatalf("Expected: %d\nGot: %d\n", arr[mid], bst.root.value)
		}
	})

	t.Run("checking parent", func(t *testing.T) {
		if bst.root.left.parent != bst.root {
			t.Fatalf("Expected: %v\nGot: %v\n", bst.root, bst.root.left.parent)
		}
		if bst.root.right.parent != bst.root {
			t.Fatalf("Expected: %v\nGot: %v\n", bst.root, bst.root.left.parent)
		}
	})
}
