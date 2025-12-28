import torch
import numpy as np
import uvicorn
import sys
import os
import traceback
import types
from collections import namedtuple, deque
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from enum import Enum

# ==========================================
# 1. CẤU HÌNH & IMPORT
# ==========================================
BLOCK_SIZE = 20
Point = namedtuple('Point', 'x, y')

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Hack module để các file solver import được
game_module = types.ModuleType("game")
game_module.Direction = Direction
game_module.Point = Point
game_module.BLOCK_SIZE = BLOCK_SIZE
sys.modules["game"] = game_module
sys.modules["snake_game"] = game_module 

try:
    from model import Linear_QNet
except ImportError:
    Linear_QNet = None

try:
    from solver_hybrid import HybridSolver
    from solver_hamilton import HamiltonSolver
    from solver_A_star import AStarSolver
except ImportError:
    pass # Bỏ qua nếu lỗi, sẽ handle ở dưới

# ==========================================
# 2. CLASS GIẢ LẬP (ĐÃ CHUẨN HÓA TỌA ĐỘ)
# ==========================================
class SimulatedGame:
    def __init__(self, snake_coords, food_coord, width_cells, height_cells):
        self.block_size = BLOCK_SIZE
        # Kích thước Pixel
        self.w = width_cells * BLOCK_SIZE
        self.h = height_cells * BLOCK_SIZE
        self.cols = width_cells
        self.rows = height_cells
        
        # --- CHỐT CỨNG: LUÔN NHÂN 20 ---
        # Giả định Frontend LUÔN gửi Grid Index (0, 1, 2...)
        # Nếu Frontend gửi Pixel (20, 40...), rắn sẽ bị phóng đại và lỗi ngay.
        # Hãy đảm bảo file compare.html và index.html gửi tọa độ chia cho 20.
        self.snake = [Point(pt[0] * BLOCK_SIZE, pt[1] * BLOCK_SIZE) for pt in snake_coords]
        self.food = Point(food_coord[0] * BLOCK_SIZE, food_coord[1] * BLOCK_SIZE)
            
        self.head = self.snake[0]
        self.score = len(self.snake) - 3
        
        # Xác định hướng
        if len(self.snake) > 1:
            neck = self.snake[1]
            if self.head.x > neck.x: self.direction = Direction.RIGHT
            elif self.head.x < neck.x: self.direction = Direction.LEFT
            elif self.head.y < neck.y: self.direction = Direction.UP # Pygame y nhỏ là trên
            elif self.head.y > neck.y: self.direction = Direction.DOWN
        else:
            self.direction = Direction.RIGHT

    def is_collision(self, pt=None):
        if pt is None: pt = self.head
        # Check tường
        if pt.x >= self.w or pt.x < 0 or pt.y >= self.h or pt.y < 0:
            return True
        # Check thân rắn
        if pt in self.snake[1:]:
            return True
        return False

    def is_trap(self, point):
        # 1. Va chạm cơ bản
        if self.is_collision(point): return True
            
        # 2. Chuẩn bị lưới ảo (Grid Index) để BFS
        start_x = int(point.x // BLOCK_SIZE)
        start_y = int(point.y // BLOCK_SIZE)
        
        tail = self.snake[-1]
        tail_x = int(tail.x // BLOCK_SIZE)
        tail_y = int(tail.y // BLOCK_SIZE)

        # Set vật cản
        obstacles = set()
        for pt in self.snake:
            ox = int(pt.x // BLOCK_SIZE)
            oy = int(pt.y // BLOCK_SIZE)
            obstacles.add((ox, oy))
        
        # Đuôi sẽ di chuyển nên không tính là vật cản
        if (tail_x, tail_y) in obstacles:
            obstacles.remove((tail_x, tail_y))
            
        if (start_x, start_y) in obstacles: return True
            
        # 3. BFS tìm đường về đuôi
        queue = deque([(start_x, start_y)])
        visited = set([(start_x, start_y)])
        
        while queue:
            cx, cy = queue.popleft()
            if cx == tail_x and cy == tail_y: return False 

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.cols and 0 <= ny < self.rows:
                    if (nx, ny) not in obstacles and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        
        return True 

# ==========================================
# 3. HÀM LẤY STATE CHO RL (CHUẨN 100% NHƯ LÚC TRAIN)
# ==========================================
def get_game_state(game):
    head = game.head
    
    # Các điểm lân cận
    point_l = Point(head.x - BLOCK_SIZE, head.y)
    point_r = Point(head.x + BLOCK_SIZE, head.y)
    point_u = Point(head.x, head.y - BLOCK_SIZE)
    point_d = Point(head.x, head.y + BLOCK_SIZE)
    
    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN

    # Tính toán Trap (Ngõ cụt)
    trap_l = game.is_trap(point_l)
    trap_r = game.is_trap(point_r)
    trap_u = game.is_trap(point_u)
    trap_d = game.is_trap(point_d)
    
    # Map Trap theo hướng nhìn của rắn
    trap_straight = (dir_r and trap_r) or (dir_l and trap_l) or (dir_u and trap_u) or (dir_d and trap_d)
    trap_right = (dir_u and trap_r) or (dir_d and trap_l) or (dir_l and trap_u) or (dir_r and trap_d)
    trap_left = (dir_d and trap_r) or (dir_u and trap_l) or (dir_r and trap_u) or (dir_l and trap_d)

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
        
        # Move direction
        dir_l, dir_r, dir_u, dir_d,
        
        # Food location
        game.food.x < game.head.x, 
        game.food.x > game.head.x, 
        game.food.y < game.head.y, 
        game.food.y > game.head.y, 
        
        # Trap info
        trap_straight,
        trap_right,
        trap_left
    ]
    return np.array(state, dtype=int)

# ==========================================
# 4. SERVER & MODEL
# ==========================================
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PICTURES_DIR = os.path.join(BASE_DIR, "pictures")
if not os.path.exists(PICTURES_DIR): os.makedirs(PICTURES_DIR)
app.mount("/pictures", StaticFiles(directory=PICTURES_DIR), name="pictures")
templates = Jinja2Templates(directory="templates")

# Load RL Model
rl_model = None
if Linear_QNet:
    try:
        rl_model = Linear_QNet(14, 256, 3)
        model_path = os.path.join(BASE_DIR, 'model', 'model.pth')
        if os.path.exists(model_path):
            rl_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            rl_model.eval()
            print("✅ RL Model Loaded")
    except: pass

class GameInput(BaseModel):
    snake: list
    food: list
    board_size: dict
    algorithm: str

# ==========================================
# 5. API PREDICT
# ==========================================
@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/compare")
def compare_page(request: Request):
    return templates.TemplateResponse("compare.html", {"request": request})

@app.post("/predict")
def predict(data: GameInput):
    game = SimulatedGame(data.snake, data.food, data.board_size['width'], data.board_size['height'])
    move_vector = [0, 0]

    try:
        # --- DEEP Q-LEARNING ---
        if data.algorithm == "rl" and rl_model:
            state = get_game_state(game)
            state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            prediction = rl_model(state0)
            move = torch.argmax(prediction).item()
            
            clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
            # Handle trường hợp game mới bắt đầu chưa có hướng
            try: idx = clock_wise.index(game.direction)
            except: idx = 0 
            
            if move == 0: new_dir = clock_wise[idx] # Thẳng
            elif move == 1: new_dir = clock_wise[(idx + 1) % 4] # Phải
            else: new_dir = clock_wise[(idx - 1) % 4] # Trái
            
            if new_dir == Direction.RIGHT: return {"move": [1, 0]}
            elif new_dir == Direction.LEFT: return {"move": [-1, 0]}
            elif new_dir == Direction.UP: return {"move": [0, -1]}
            elif new_dir == Direction.DOWN: return {"move": [0, 1]}

        # --- CLASSIC ALGORITHMS ---
        else:
            solver = None
            if data.algorithm == "astar":
                solver = AStarSolver(game)
            elif data.algorithm == "hybrid":
                solver = HybridSolver(game)
            elif data.algorithm == "hamilton":
                solver = HamiltonSolver(game)

            next_pt = solver.get_next_move() if solver else None

            if next_pt:
                dx = int((next_pt.x - game.head.x) / BLOCK_SIZE)
                dy = int((next_pt.y - game.head.y) / BLOCK_SIZE)
                move_vector = [dx, dy]
            else:
                # Fallback: Đi theo hướng hiện tại nếu không tìm thấy đường
                if game.direction == Direction.RIGHT: move_vector = [1, 0]
                elif game.direction == Direction.LEFT: move_vector = [-1, 0]
                elif game.direction == Direction.UP: move_vector = [0, -1]
                elif game.direction == Direction.DOWN: move_vector = [0, 1]

    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()

    return {"move": move_vector}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)