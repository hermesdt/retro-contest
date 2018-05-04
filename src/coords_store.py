from collections import deque
import math


class CoordsStore():
    def __init__(self, maxlen=4, min_distance_allowed=40):
        self.coords = deque(maxlen=maxlen)
        self.min_distance_allowed = min_distance_allowed

    def add(self, x, y):
        smallest_distance = self.smallest_distance(x, y)
        print(smallest_distance)
        if smallest_distance > self.min_distance_allowed:
            self.coords.append([x, y])

    def smallest_distance(self, x, y):
        distance = min([math.sqrt((a - x) ** 2 + (b - y) ** 2) for a, b in self.coords] or [5_000])
        return distance


if __name__ == "__main__":
    s = CoordsStore()
    print(s.coords)
    print(s.smallest_distance(0, 1))