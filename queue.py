class Node:
    def __init__(self, value, nxt=None):
        self.value = value
        self.next = nxt

    def set_next(self, nxt):
        self.next = nxt

    def get_next(self):
        return self.next

    def get_value(self):
        return self.next


class Queue:
    def __init__(self):
        self.head = None
        self.tail = None

    def is_empty(self):
        return self.head is None

    def push(self, element):
        new_tail = Node(element)
        if self.is_empty():
            self.head = new_tail
            self.tail = new_tail
        else:
            self.tail.set_next(new_tail)
            self.tail = new_tail

    def pop(self):
        if self.is_empty():
            return Exception('Queue has no elements')
        old_head = self.head
        self.head = self.head.get_next()
        return old_head.value
