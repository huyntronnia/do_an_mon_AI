import sys
import os
import traceback
from collections import namedtuple
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles 
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from enum import Enum
from unittest.mock import MagicMock

# ==========================================
# 1. HACK: GIẢ LẬP PYGAME (ĐỂ THUẬT TOÁN KHÔNG BỊ LỖI IMPORT)
# ==========================================
# Code này lừa các file solver rằng pygame đã được cài đặt
try:
    import pygame
except ImportError:
    mock = MagicMock()
    mock.init.return_value = True
    sys.modules["pygame"] = mock
    sys.modules["pygame.locals"] = MagicMock()
    sys.modules["pygame.time"] = MagicMock()
    sys.modules["pygame.display"] = MagicMock()
    sys.modules["pygame.event"] = MagicMock()
    sys.modules["pygame.draw"] = MagicMock()
    sys.modules["pygame.font"] = MagicMock()

import torch
import numpy as np
import uvicorn

# ==========================================
# 2. CẤU HÌNH ĐƯỜNG DẪN & IMPORT
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PICTURES_DIR = os.path.join(BASE_DIR, "pictures")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
sys.path.append(BASE_DIR) # Để tìm thấy file helper.py

# Cố gắng import Point từ helper để đồng bộ dữ liệu với thuật toán
try:
    from helper import Point
except ImportError:
    try:
        from game import Point
    except ImportError:
        Point = namedtuple('Point', 'x, y')

try:
    from game import Direction
except ImportError:
    class Direction(Enum):
        RIGHT = 1
        LEFT = 2
        UP = 3
        DOWN = 4

# Import Thuật toán
try:
    from model import Linear_QNet
except ImportError:
    Linear_QNet = None

try:
    from solver_hybrid import HybridSolver
    from solver_hamilton import HamiltonSolver
    from solver_A_star import AStarSolver
except ImportError:
    print("❌ Lỗi import Solver (Dù đã hack pygame). Kiểm tra lại tên file.")

BLOCK_SIZE = 20

# ==========================================
# 3. CLASS SIMULATED GAME (LOGIC CHUẨN)
# ==========================================
class SimulatedGame:
    def __init__(self, snake_coords, food_coord, width, height):
        self.block_size = BLOCK_SIZE
        self.w = width * BLOCK_SIZE
        self.h = height * BLOCK_SIZE
        self.cols = width
        self.rows = height
        self.score = 0
        
        # --- QUY ĐỔI GRID -> PIXEL ---
        # Đây là lý do code cũ chạy được: Nó nhân 20 vào tọa độ
        self.snake = []
        for pt in snake_coords:
            self.snake.append(Point(pt[0] * BLOCK_SIZE, pt[1] * BLOCK_SIZE))
            
        self.food = Point(food_coord[0] * BLOCK_SIZE, food_coord[1] * BLOCK_SIZE)
        self.head = self.snake[0]
        self.score = len(self.snake) - 3
        
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
    
    def reset(self): pass

# ==========================================
# 4. SERVER SETUP (CÓ FONT)
# ==========================================
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- DÒNG NÀY ĐỂ HIỆN FONT AKIRA ---
if not os.path.exists(PICTURES_DIR): os.makedirs(PICTURES_DIR)
app.mount("/pictures", StaticFiles(directory=PICTURES_DIR), name="pictures")

templates = Jinja2Templates(directory=TEMPLATES_DIR)

# RL State Logic
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
        (dir_r and game.is_collision(point_r)) or (dir_l and game.is_collision(point_l)) or (dir_u and game.is_collision(point_u)) or (dir_d and game.is_collision(point_d)),
        (dir_u and game.is_collision(point_r)) or (dir_d and game.is_collision(point_l)) or (dir_l and game.is_collision(point_u)) or (dir_r and game.is_collision(point_d)),
        (dir_d and game.is_collision(point_r)) or (dir_u and game.is_collision(point_l)) or (dir_r and game.is_collision(point_u)) or (dir_l and game.is_collision(point_d)),
        dir_l, dir_r, dir_u, dir_d,
        game.food.x < game.head.x, game.food.x > game.head.x, game.food.y < game.head.y, game.food.y > game.head.y,
        0, 0, 0 
    ]
    return np.array(state, dtype=int)

rl_model = None
if Linear_QNet:
    try:
        rl_model = Linear_QNet(14, 256, 3)
        model_path = os.path.join(BASE_DIR, 'model', 'model.pth')
        if os.path.exists(model_path):
            rl_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            rl_model.eval()
            print("✅ RL Model Loaded")
    except Exception: pass

class GameInput(BaseModel):
    snake: list
    food: list
    board_size: dict
    algorithm: str

# ==========================================
# 5. API
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
    next_pt = None

    try:
        if data.algorithm == "rl" and rl_model:
            state = get_game_state(game)
            state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            prediction = rl_model(state0)
            move = torch.argmax(prediction).item()
            clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
            try: idx = clock_wise.index(game.direction)
            except: idx = 0 
            if move == 0: new_dir = clock_wise[idx] 
            elif move == 1: new_dir = clock_wise[(idx + 1) % 4] 
            else: new_dir = clock_wise[(idx - 1) % 4] 
            
            if new_dir == Direction.RIGHT: return {"move": [1, 0]}
            elif new_dir == Direction.LEFT: return {"move": [-1, 0]}
            elif new_dir == Direction.UP: return {"move": [0, -1]}
            elif new_dir == Direction.DOWN: return {"move": [0, 1]}

        else:
            # Search Algos
            solver = None
            if data.algorithm == "astar":
                # Thử khởi tạo, nếu lỗi thì bỏ qua
                try: solver = AStarSolver(game)
                except NameError: pass
            elif data.algorithm == "hybrid":
                try: solver = HybridSolver(game)
                except NameError: pass
            elif data.algorithm == "hamilton":
                try: solver = HamiltonSolver(game)
                except NameError: pass

            if solver:
                next_pt = solver.get_next_move()

            if next_pt:
                dx = int((next_pt.x - game.head.x) / BLOCK_SIZE)
                dy = int((next_pt.y - game.head.y) / BLOCK_SIZE)
                move_vector = [dx, dy]
            else:
                # Fallback đi thẳng
                if game.direction == Direction.RIGHT: move_vector = [1, 0]
                elif game.direction == Direction.LEFT: move_vector = [-1, 0]
                elif game.direction == Direction.UP: move_vector = [0, -1]
                elif game.direction == Direction.DOWN: move_vector = [0, 1]

    except Exception as e:
        print(f"Error: {e}")
        move_vector = [1, 0]

    return {"move": move_vector}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)