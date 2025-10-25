// Linked list implementation of a Stack.

const std = @import("std");

const Node = struct {
    value: i32,
    next: ?*Node = null,
};

const Stack = struct {
    allocator: std.mem.Allocator,
    top: ?*Node,

    pub fn init(allocator: std.mem.Allocator) Stack {
        return Stack{
            .allocator = allocator,
            .top = null,
        };
    }

    pub fn push(self: *Stack, value: i32) !void {
        const node = try self.allocator.create(Node);
        node.value = value;
        node.next = self.top;
        self.top = node;
    }

    pub fn pop(self: *Stack) ?i32 {
        if (self.top) |top| {
            self.top = top.next;
            const value = top.value;
            self.allocator.destroy(top);
            return value;
        }
        return null;
    }
};

test "stack" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var stack = Stack.init(allocator);

    try std.testing.expectEqual(stack.pop(), null);

    try stack.push(4);
    try stack.push(3);
    try stack.push(2);
    try stack.push(1);

    try std.testing.expectEqual(stack.pop(), 1);
    try std.testing.expectEqual(stack.pop(), 2);
    try std.testing.expectEqual(stack.pop(), 3);
    try std.testing.expectEqual(stack.pop(), 4);

    try std.testing.expectEqual(stack.pop(), null);

    try stack.push(5);
    try std.testing.expectEqual(stack.pop(), 5);

    try std.testing.expectEqual(stack.pop(), null);
}
