import heapq
from collections import deque
from snake_game import Point, BLOCK_SIZE
import random
import math

class Node:
   
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position
    
    def __lt__(self, other):
        return self.f < other.f

class HybridSolver:
    def __init__(self, game):
        self.game = game
        self.cols = game.w // BLOCK_SIZE
        self.rows = game.h // BLOCK_SIZE

    def get_distance(self, start, end):
        # Dùng khoảng cách Euclid để ước lượng chính xác hơn về hướng
        return math.sqrt((start.x - end.x)**2 + (start.y - end.y)**2)

    def get_manhattan_distance(self, start, end):
        return abs(start.x - end.x) + abs(start.y - end.y)

    def get_neighbors(self, head):
        neighbors = []
        # Random để rắn bớt bị kẹt vào các pattern lặp lại
        directions = [(0, -BLOCK_SIZE), (0, BLOCK_SIZE), (-BLOCK_SIZE, 0), (BLOCK_SIZE, 0)]
        random.shuffle(directions)
        
        for dx, dy in directions:
            new_pt = Point(head.x + dx, head.y + dy)
            if 0 <= new_pt.x < self.game.w and 0 <= new_pt.y < self.game.h:
                neighbors.append(new_pt)
        return neighbors

    def a_star_path(self, start, target, obstacles):
        
        obstacles_set = set(obstacles)
        if start in obstacles_set: return None
        if start == target: return []

        start_node = Node(None, start)
        end_node = Node(None, target)
        
        open_list = []
        closed_list = set()
        
        heapq.heappush(open_list, start_node)
        
        # Giới hạn tìm kiếm thấp để phản ứng nhanh
        steps = 0
        max_steps = 2500 

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

    def get_next_move(self):
       
        start = self.game.head
        food = self.game.food
        snake = self.game.snake
        
        # 1. Tập vật cản cơ bản (thân rắn trừ đuôi)
        obstacles = set(snake[:-1]) 
        
        # --- PHASE 1: TÌM ĐƯỜNG ĐẾN THỨC ĂN ---
        path_to_food = self.a_star_path(start, food, obstacles)

        if path_to_food and len(path_to_food) > 1:
            next_move = path_to_food[1]
            
            # GIẢ LẬP: Nếu ăn mồi xong (rắn dài ra), đuôi vẫn ở chỗ cũ
            virtual_snake_body = [next_move] + snake[:] 
            virtual_tail = virtual_snake_body[-1] # Đuôi chưa di chuyển
            virtual_obstacles = set(virtual_snake_body[:-1])

            # Kiểm tra 1: Có đường về đuôi không?
            path_back = self.a_star_path(next_move, virtual_tail, virtual_obstacles)
            
            if path_back:
                return next_move # An toàn tuyệt đối -> ĂN!

            # Kiểm tra 2 (QUAN TRỌNG ĐỂ PHÁ VÒNG LẶP):
            # Nếu không về được đuôi, nhưng vùng không gian rất rộng (>70% ô trống)
            # Thì vẫn cứ ăn! Rủi ro thấp.
            total_cells = self.cols * self.rows
            empty_cells = total_cells - len(snake)
            
            # Đếm vùng không gian tại ô dự định đi
            space = self.bfs_flood_fill(next_move, virtual_obstacles)
            
            # Nếu vùng không gian chiếm > 70% số ô trống còn lại -> An toàn tương đối
            if space > empty_cells * 0.7:
                 return next_move

        # --- PHASE 2: KHÔNG ĂN ĐƯỢC -> TÌM Ô "TỐT NHẤT" ĐỂ ĐI ---
        # Mục tiêu: Sống sót và Lại gần mồi hơn (nếu có thể)
        
        neighbors = self.get_neighbors(start)
        best_move = None
        max_score = -999999
        
        tail = snake[-1]
        # Khi di chuyển bình thường, đuôi sẽ chạy theo -> vật cản bớt đi 1 đốt đuôi
        obstacles_move = set(snake[:-1]) 

        for move in neighbors:
            if move in obstacles_move: continue
            
            # Tính điểm cho nước đi này
            score = 0
            
            # 1. Kiểm tra an toàn (quan trọng nhất): Phải về được đuôi
            # Giả lập di chuyển (đuôi chạy đi)
            path_to_tail = self.a_star_path(move, tail, obstacles_move)
            
            if path_to_tail:
                score += 10000 # Điểm cộng cực lớn cho sự an toàn
                
                # 2. Ưu tiên ô có không gian rộng (Flood fill)
                # Để tránh chui vào ngõ cụt
                space = self.bfs_flood_fill(move, obstacles_move)
                score += space * 10
                
                # 3. Ưu tiên ô GẦN THỨC ĂN HƠN (Hungry Strategy)
                # Đây là dòng giúp phá vỡ việc chạy quanh biên
                dist = self.get_distance(move, food)
                score -= dist * 5 # Khoảng cách càng nhỏ điểm càng cao
            else:
                # Nếu không về được đuôi, coi như nước đi tử thần, trừ điểm nặng
                score -= 10000
                # Nhưng nếu buộc phải đi (không còn đường nào), chọn ô có không gian rộng nhất
                space = self.bfs_flood_fill(move, obstacles_move)
                score += space

            if score > max_score:
                max_score = score
                best_move = move
        
        return best_move