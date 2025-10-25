// Binary Search Tree implementation

package main

type TreeNode struct {
	value  int
	left   *TreeNode
	right  *TreeNode
	parent *TreeNode
}

type BinarySearchTree struct {
	root *TreeNode
}

func (n *TreeNode) insert(data int) {
	if n.value == data {
		return
	}

	if n.value < data {
		if n.right != nil {
			n.right.insert(data)
		} else {
			n.right = &TreeNode{value: data}
			n.right.parent = n
		}
	} else {
		if n.left != nil {
			n.left.insert(data)
		} else {
			n.left = &TreeNode{value: data}
			n.left.parent = n
		}
	}
}

func (n *TreeNode) search(target int) *TreeNode {
	if n.value == target {
		return n
	}
	if n.value < target && n.right != nil {
		return n.right.search(target)
	}
	if n.value > target && n.left != nil {
		return n.left.search(target)
	}
	return nil
}

func (n *TreeNode) traverseInorder() []int {
	res := []int{}
	if n != nil {
		res = append(res, n.left.traverseInorder()...)
		res = append(res, n.value)
		res = append(res, n.right.traverseInorder()...)
	}
	return res
}

func (t *BinarySearchTree) Add(data int) {
	if t.root == nil {
		t.root = &TreeNode{value: data}
	} else {
		t.root.insert(data)
	}

}
func (t *BinarySearchTree) Remove(node *TreeNode) {
	if t.root == nil || node == nil {
		return
	}

	// Node with no children
	if node.left == nil && node.right == nil {
		if node.parent == nil {
			t.root = nil
		} else if node.parent.left == node {
			node.parent.left = nil
		} else {
			node.parent.right = nil
		}
		node.parent = nil
		return
	}

	// Node with one child
	if node.left == nil || node.right == nil {
		child := node.left
		if node.right != nil {
			child = node.right
		}
		child.parent = node.parent
		if node.parent == nil {
			t.root = child
		} else if node.parent.left == node {
			node.parent.left = child
		} else {
			node.parent.right = child
		}
		node.parent = nil
		node.left = nil
		node.right = nil
		return
	}

	// Node with two children
	successor := node.right
	for successor.left != nil {
		successor = successor.left
	}
	t.Remove(successor)
	node.value = successor.value
}

func (t *BinarySearchTree) Search(target int) *TreeNode {
	if t.root == nil {
		return nil
	}
	return t.root.search(target)
}

func (t *BinarySearchTree) TraverseInorder() []int {
	return t.root.traverseInorder()
}

func buildTree(parent *TreeNode, arr []int, low, high int) *TreeNode {
	if high < low {
		return nil
	}
	mid := (low + high) / 2
	node := &TreeNode{value: arr[mid]}
	node.left = buildTree(node, arr, low, mid-1)
	node.right = buildTree(node, arr, mid+1, high)
	node.parent = parent
	return node
}

func BulkConstruct(arr []int) *BinarySearchTree {
	return &BinarySearchTree{root: buildTree(nil, arr, 0, len(arr)-1)}
}
