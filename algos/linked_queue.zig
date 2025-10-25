// Linked list implementation of a Queue.

const std = @import("std");

const Node = struct {
    value: i32,
    next: ?*Node = null,
};

const Queue = struct {
    allocator: std.mem.Allocator,
    head: ?*Node,
    tail: ?*Node,

    pub fn init(allocator: std.mem.Allocator) Queue {
        return Queue{
            .allocator = allocator,
            .head = null,
            .tail = null,
        };
    }

    pub fn enqueue(self: *Queue, value: i32) !void {
        const node = try self.allocator.create(Node);
        node.value = value;
        node.next = null;
        if (self.tail) |tail| {
            tail.next = node;
        } else {
            self.head = node;
        }
        self.tail = node;
    }

    pub fn dequeue(self: *Queue) ?i32 {
        if (self.head) |head| {
            self.head = head.next;
            if (head.next == null) {
                self.tail = null;
            }
            const value = head.value;
            self.allocator.destroy(head);
            return value;
        }
        return null;
    }
};

test "queue" {
    var queue = Queue.init(std.testing.allocator);

    try std.testing.expectEqual(queue.dequeue(), null);

    try queue.enqueue(1);
    try queue.enqueue(2);
    try queue.enqueue(3);
    try queue.enqueue(4);

    try std.testing.expectEqual(queue.dequeue(), 1);
    try std.testing.expectEqual(queue.dequeue(), 2);
    try std.testing.expectEqual(queue.dequeue(), 3);
    try std.testing.expectEqual(queue.dequeue(), 4);

    try queue.enqueue(5);
    try std.testing.expectEqual(queue.dequeue(), 5);

    try std.testing.expectEqual(queue.dequeue(), null);
}
