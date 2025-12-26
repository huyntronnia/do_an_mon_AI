import heapq
from collections import deque
import random
import math
from snake_game import Point, BLOCK_SIZE

class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0
    def __eq__(self, other): return self.position == other.position
    def __lt__(self, other): return self.f < other.f

class BaseSolver:
    def __init__(self, game):
        self.game = game
        self.cols = game.w // BLOCK_SIZE
        self.rows = game.h // BLOCK_SIZE

    def get_distance(self, start, end):
        return math.sqrt((start.x - end.x)**2 + (start.y - end.y)**2)

    def get_manhattan_distance(self, start, end):
        return abs(start.x - end.x) + abs(start.y - end.y)

    def get_neighbors(self, head):
        neighbors = []
        directions = [(0, -BLOCK_SIZE), (0, BLOCK_SIZE), (-BLOCK_SIZE, 0), (BLOCK_SIZE, 0)]
        random.shuffle(directions)
        for dx, dy in directions:
            new_pt = Point(head.x + dx, head.y + dy)
            if 0 <= new_pt.x < self.game.w and 0 <= new_pt.y < self.game.h:
                neighbors.append(new_pt)
        return neighbors

    def a_star_path(self, start, target, obstacles):
        """Tìm đường đi ngắn nhất (Dùng chung cho cả A* và Hybrid)"""
        obstacles_set = set(obstacles)
        if start in obstacles_set: return None
        if start == target: return []

        start_node = Node(None, start)
        end_node = Node(None, target)
        
        open_list = []
        closed_list = set()
        
        heapq.heappush(open_list, start_node)
        steps = 0
        
        # Giới hạn bước tìm kiếm để game không bị lag
        max_steps = 3000 

        while open_list and steps < max_steps:
            steps += 1
            current_node = heapq.heappop(open_list)
            closed_list.add(current_node.position)

            if current_node == end_node:
                path = []
                curr = current_node
                while curr is not None:
                    path.append(curr.position)
                    curr = curr.parent
                return path[::-1]

            for neighbor_pos in self.get_neighbors(current_node.position):
                if neighbor_pos in obstacles_set or neighbor_pos in closed_list:
                    continue

                new_node = Node(current_node, neighbor_pos)
                new_node.g = current_node.g + 1
                new_node.h = self.get_manhattan_distance(new_node.position, end_node.position)
                new_node.f = new_node.g + new_node.h
                heapq.heappush(open_list, new_node)
        return None

    def bfs_flood_fill(self, start, obstacles):
        """Đếm không gian trống"""
        obstacles_set = set(obstacles)
        if start in obstacles_set: return 0
        queue = deque([start])
        visited = set([start])
        count = 0
        limit = 300
        while queue:
            current = queue.popleft()
            count += 1
            if count >= limit: return count
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited and neighbor not in obstacles_set:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return count