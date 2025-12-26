import torch
import random
import numpy as np
import heapq
import math
import uvicorn
import sys
import os
from collections import namedtuple, deque
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from enum import Enum

# ==========================================
# 1. IMPORT CÁC MODULE PHỤ TRỢ
# ==========================================
# Cố gắng import Agent và Model, nếu thiếu thì báo lỗi nhẹ để code vẫn chạy phần khác
try:
    from model import Linear_QNet
except ImportError:
    print("⚠️ Cảnh báo: Không tìm thấy file model.py. Deep Q-Learning sẽ không hoạt động.")
    Linear_QNet = None

# ==========================================
# 2. CẤU HÌNH & ĐỊNH NGHĨA CƠ BẢN
# ==========================================
BLOCK_SIZE = 20
Point = namedtuple('Point', 'x, y')

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# ==========================================
# 3. THUẬT TOÁN: HYBRID A* & FLOOD FILL
# ==========================================
class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0
    def __eq__(self, other): return self.position == other.position
    def __lt__(self, other): return self.f < other.f

class HybridSolver:
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
        obstacles_set = set(obstacles)
        if start in obstacles_set: obstacles_set.remove(start)
        if start == target: return []
        
        start_node = Node(None, start)
        end_node = Node(None, target)
        open_list = []
        closed_list = set()
        heapq.heappush(open_list, start_node)
        
        steps = 0
        while open_list and steps < 2000: # Limit steps to prevent lag
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
                if neighbor_pos in obstacles_set or neighbor_pos in closed_list: continue
                new_node = Node(current_node, neighbor_pos)
                new_node.g = current_node.g + 1
                new_node.h = self.get_manhattan_distance(new_node.position, end_node.position)
                new_node.f = new_node.g + new_node.h
                heapq.heappush(open_list, new_node)
        return None

    def bfs_flood_fill(self, start, obstacles):
        obstacles_set = set(obstacles)
        if start in obstacles_set: obstacles_set.remove(start)
        queue = deque([start])
        visited = set([start])
        count = 0
        while queue:
            current = queue.popleft()
            count += 1
            if count >= 200: return count # Limit flood fill
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited and neighbor not in obstacles_set:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return count

    def get_next_move(self):
        start = self.game.head
        food = self.game.food
        snake = self.game.snake
        obstacles = set(snake)
        if snake[-1] in obstacles: obstacles.remove(snake[-1])
        
        # 1. Thử tìm đường ngắn nhất đến thức ăn
        path_to_food = self.a_star_path(start, food, obstacles)
        
        if path_to_food and len(path_to_food) > 1:
            next_move = path_to_food[1]
            # Kiểm tra xem đi bước này có bị kẹt vào ngõ cụt không (Flood Fill)
            virtual_snake = [next_move] + snake[:-1]
            virtual_obstacles = set(virtual_snake)
            if virtual_snake[-1] in virtual_obstacles: virtual_obstacles.remove(virtual_snake[-1])
            
            # Nếu đường còn thông thoáng (có thể tìm đường về đuôi hoặc khoảng trống lớn)
            if self.a_star_path(next_move, virtual_snake[-1], virtual_obstacles): return next_move
            space = self.bfs_flood_fill(next_move, virtual_obstacles)
            if space > ((self.game.w * self.game.h) // (BLOCK_SIZE**2) - len(snake)) * 0.5: return next_move

        # 2. Nếu không có đường an toàn đến thức ăn, tìm nước đi sống sót lâu nhất
        neighbors = self.get_neighbors(start)
        best_move = None
        max_score = -float('inf')
        
        obstacles_safety = set(snake)
        if snake[-1] in obstacles_safety: obstacles_safety.remove(snake[-1])
        
        for move in neighbors:
            if move in obstacles_safety: continue
            score = 0
            # Ưu tiên đi được về phía đuôi
            if self.a_star_path(move, snake[-1], obstacles_safety): 
                score += 5000 - self.get_distance(move, food)
            else: 
                # Nếu không thì chọn vùng rộng nhất
                score += self.bfs_flood_fill(move, obstacles_safety) * 10 - 1000
            
            if score > max_score:
                max_score = score
                best_move = move
        return best_move

# ==========================================
# 4. THUẬT TOÁN: HAMILTONIAN CYCLE (ĐÃ KHÔI PHỤC FULL CODE)
# ==========================================
class HamiltonSolver:
    def __init__(self, game):
        self.game = game
        self.hamiltonian_path = self._build_hamiltonian_cycle()

    def _build_hamiltonian_cycle(self):
        """Tạo chu trình zig-zag phủ kín bản đồ"""
        path_map = {}
        # Duyệt qua các cột
        for x in range(0, self.game.w, BLOCK_SIZE):
            col_idx = x // BLOCK_SIZE
            
            # Cột chẵn: Đi xuống
            if col_idx % 2 == 0:
                for y in range(BLOCK_SIZE, self.game.h - BLOCK_SIZE, BLOCK_SIZE):
                    path_map[Point(x, y)] = Point(x, y + BLOCK_SIZE)
                path_map[Point(x, self.game.h - BLOCK_SIZE)] = Point(x + BLOCK_SIZE, self.game.h - BLOCK_SIZE)
            
            # Cột lẻ: Đi lên
            else:
                for y in range(self.game.h - BLOCK_SIZE, BLOCK_SIZE, -BLOCK_SIZE):
                    path_map[Point(x, y)] = Point(x, y - BLOCK_SIZE)
                
                if col_idx < (self.game.w // BLOCK_SIZE) - 1:
                    path_map[Point(x, BLOCK_SIZE)] = Point(x + BLOCK_SIZE, BLOCK_SIZE)
                else:
                    path_map[Point(x, BLOCK_SIZE)] = Point(x, 0) # Về đích

        # Hàng ngang trên cùng về gốc
        for x in range(self.game.w - BLOCK_SIZE, 0, -BLOCK_SIZE):
            path_map[Point(x, 0)] = Point(x - BLOCK_SIZE, 0)
        
        path_map[Point(0, 0)] = Point(0, BLOCK_SIZE)
        return path_map

    def get_next_move(self):
        if self.game.head in self.hamiltonian_path:
            return self.hamiltonian_path[self.game.head]
        
        # Fallback an toàn nếu bị lệch khỏi đường
        for dx, dy in [(20,0), (-20,0), (0,20), (0,-20)]:
             if 0 <= self.game.head.x+dx < self.game.w and 0 <= self.game.head.y+dy < self.game.h:
                 return Point(self.game.head.x+dx, self.game.head.y+dy)
        return None

# ==========================================
# 5. SETUP SERVER & SIMULATED GAME & RL MODEL
# ==========================================
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="templates"), name="static")
templates = Jinja2Templates(directory="templates")

# --- LOAD RL MODEL ---
rl_model = None
if Linear_QNet:
    try:
        # Input size = 14 (Khớp với logic get_game_state bên dưới)
        rl_model = Linear_QNet(14, 256, 3)
        model_path = './model/model.pth'
        if os.path.exists(model_path):
            rl_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            rl_model.eval()
            print("✅ RL Model Loaded Successfully (CPU Mode)")
        else:
            print("❌ Không tìm thấy file ./model/model.pth")
    except Exception as e:
        print(f"❌ Lỗi khi load Model: {e}")

class SimulatedGame:
    def __init__(self, snake_coords, food_coord, width, height):
        self.block_size = BLOCK_SIZE
        # Quy đổi kích thước Grid -> Pixel
        self.w = width * BLOCK_SIZE
        self.h = height * BLOCK_SIZE
        
        # --- FIX: Tự động nhận diện toạ độ Pixel hay Grid ---
        sample_x = snake_coords[0][0]
        is_pixel_coords = sample_x > 40 # Nếu toạ độ > 40, giả định là Pixel
        
        if is_pixel_coords:
            self.snake = [Point(x, y) for x, y in snake_coords]
            self.food = Point(food_coord[0], food_coord[1])
        else:
            self.snake = [Point(x * BLOCK_SIZE, y * BLOCK_SIZE) for x, y in snake_coords]
            self.food = Point(food_coord[0] * BLOCK_SIZE, food_coord[1] * BLOCK_SIZE)
            
        self.head = self.snake[0]
        
        # Xác định hướng
        if len(self.snake) > 1:
            neck = self.snake[1]
            if self.head.x > neck.x: self.direction = Direction.RIGHT
            elif self.head.x < neck.x: self.direction = Direction.LEFT
            elif self.head.y < neck.y: self.direction = Direction.UP
            elif self.head.y > neck.y: self.direction = Direction.DOWN
        else:
            self.direction = Direction.RIGHT

    def is_collision(self, pt=None):
        if pt is None: pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

# --- HÀM TÍNH TRẠNG THÁI CHO RL (INPUT: 14) ---
def get_game_state(game):
    head = game.head
    point_l = Point(head.x - 20, head.y)
    point_r = Point(head.x + 20, head.y)
    point_u = Point(head.x, head.y - 20)
    point_d = Point(head.x, head.y + 20)
    
    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN

    state = [
        # Danger Straight
        (dir_r and game.is_collision(point_r)) or 
        (dir_l and game.is_collision(point_l)) or 
        (dir_u and game.is_collision(point_u)) or 
        (dir_d and game.is_collision(point_d)),

        # Danger Right
        (dir_u and game.is_collision(point_r)) or 
        (dir_d and game.is_collision(point_l)) or 
        (dir_l and game.is_collision(point_u)) or 
        (dir_r and game.is_collision(point_d)),

        # Danger Left
        (dir_d and game.is_collision(point_r)) or 
        (dir_u and game.is_collision(point_l)) or 
        (dir_r and game.is_collision(point_u)) or 
        (dir_l and game.is_collision(point_d)),
        
        # Move Direction
        dir_l, dir_r, dir_u, dir_d,
        
        # Food Location 
        game.food.x < game.head.x,  # Food Left
        game.food.x > game.head.x,  # Food Right
        game.food.y < game.head.y,  # Food Up
        game.food.y > game.head.y,  # Food Down
        
        # Extra inputs (FloodFill/Safety) - Giữ chỗ cho đủ 14 input
        0, 0, 0
    ]
    return np.array(state, dtype=int)

class GameInput(BaseModel):
    snake: list
    food: list
    board_size: dict
    algorithm: str

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict(data: GameInput):
    game = SimulatedGame(data.snake, data.food, data.board_size['width'], data.board_size['height'])
    move_vector = [0, 0]

    try:
        # --- XỬ LÝ: A* & HYBRID ---
        if data.algorithm in ["hybrid", "astar"]:
            solver = HybridSolver(game)
            if data.algorithm == "astar":
                obstacles = set(game.snake)
                if game.snake[0] in obstacles: obstacles.remove(game.snake[0])
                if game.snake[-1] in obstacles: obstacles.remove(game.snake[-1])
                path = solver.a_star_path(game.head, game.food, obstacles)
                next_pt = path[1] if path and len(path) > 1 else None
            else:
                next_pt = solver.get_next_move()
            
            if next_pt:
                move_vector = [int((next_pt.x - game.head.x)/BLOCK_SIZE), int((next_pt.y - game.head.y)/BLOCK_SIZE)]

        # --- XỬ LÝ: HAMILTONIAN ---
        elif data.algorithm == "hamilton":
            solver = HamiltonSolver(game)
            next_pt = solver.get_next_move()
            if next_pt:
                move_vector = [int((next_pt.x - game.head.x)/BLOCK_SIZE), int((next_pt.y - game.head.y)/BLOCK_SIZE)]

        # --- XỬ LÝ: DEEP Q-LEARNING ---
        elif data.algorithm == "rl" and rl_model:
            # 1. Lấy trạng thái
            state = get_game_state(game)
            
            # 2. Convert to Tensor (Thêm chiều batch [1, 14])
            state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            
            # 3. Predict
            prediction = rl_model(state0)
            move = torch.argmax(prediction).item() # 0: Straight, 1: Right, 2: Left
            
            # 4. Map Move to Vector
            clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
            idx = clock_wise.index(game.direction)
            
            if move == 0: # Straight
                new_dir = clock_wise[idx]
            elif move == 1: # Right Turn
                new_dir = clock_wise[(idx + 1) % 4]
            else: # Left Turn
                new_dir = clock_wise[(idx - 1) % 4]

            if new_dir == Direction.RIGHT: move_vector = [1, 0]
            elif new_dir == Direction.LEFT: move_vector = [-1, 0]
            elif new_dir == Direction.UP: move_vector = [0, -1]
            elif new_dir == Direction.DOWN: move_vector = [0, 1]

    except Exception as e:
        print(f"Lỗi Xử Lý: {e}")
        # Fallback: Đi sang phải nếu lỗi
        move_vector = [1, 0]

    return {"move": move_vector}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)