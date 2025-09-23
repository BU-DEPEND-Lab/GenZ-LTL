import heapq
from typing import TypeVar, Generic, Optional

T = TypeVar('T')  # Generic type for items in the priority queue


class PriorityQueue(Generic[T]):
    def __init__(self):
        self._heap: list[list[int | Optional[T]]] = []  # optional to support removing items; list to modify in place
        self._entries: dict[T, list[int | Optional[T]]] = {}  # maps items to entries

    def push(self, item: T, priority: int) -> None:
        """
        Push an item onto the priority queue with a given priority.
        Lower values indicate higher priority.
        """
        entry = [priority, item]
        self._entries[item] = entry
        heapq.heappush(self._heap, entry)

    def change_priority(self, item: T, priority: int) -> None:
        """
        Change the priority of an item in the priority queue.
        """
        if item not in self._entries:
            raise ValueError(f"Item {item} not in priority queue")
        entry = self._entries.pop(item)
        entry[-1] = None
        new_entry = [priority, item]
        self._entries[item] = new_entry
        heapq.heappush(self._heap, new_entry)

    def pop(self) -> T:
        """
        Remove and return the item with the highest priority (lowest priority number).
        """
        while self._heap:
            [priority, item] = heapq.heappop(self._heap)
            if item is not None:
                del self._entries[item]
                return item, priority
        raise IndexError("pop from an empty priority queue")

    def is_empty(self) -> bool:
        """
        Return True if the priority queue is empty, False otherwise.
        """
        return len(self._heap) == 0

    def __len__(self) -> int:
        """
        Return the number of items in the priority queue.
        """
        return len(self._heap)

    def __contains__(self, item: T) -> bool:
        """
        Return True if the item is in the priority queue, False otherwise.
        """
        return item in self._entries

    def __getitem__(self, item) -> int:
        """
        Return the priority of an item in the priority queue.
        """
        entry = self._entries.get(item)
        if entry is None or entry[1] is None:
            raise KeyError(f"Item {item} not in priority queue")
        return entry[0]
