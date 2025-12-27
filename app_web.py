import torch
import random
import numpy as np
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
# 1. IMPORT CÁC MODULE THUẬT TOÁN (QUAN TRỌNG)
# ==========================================
try:
    from model import Linear_QNet
except ImportError:
    print("⚠️ Cảnh báo: Không tìm thấy file model.py. Deep Q-Learning sẽ không hoạt động.")
    Linear_QNet = None

# Import trực tiếp từ các file solver bạn đã viết
# Đảm bảo các file này nằm cùng thư mục với app_web.py
try:
    from solver_hybrid import HybridSolver
    from solver_hamilton import HamiltonSolver
    from solver_A_star import AStarSolver
except ImportError as e:
    print(f"⚠️ Lỗi Import Solver: {e}")
    print("Hãy chắc chắn rằng các file solver_*.py và utils.py nằm cùng thư mục.")

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
# 3. CLASS GIẢ LẬP GAME (ADAPTER)
# ==========================================
# Class này đóng vai trò cầu nối để các Solver (vốn viết cho pygame)
# có thể hiểu được dữ liệu từ Web gửi lên.
class SimulatedGame:
    def __init__(self, snake_coords, food_coord, width, height):
        self.block_size = BLOCK_SIZE
        # Web gửi width/height là số ô (ví dụ 32, 24), cần nhân với BLOCK_SIZE
        self.w = width * BLOCK_SIZE
        self.h = height * BLOCK_SIZE
        
        # Kiểm tra xem dữ liệu gửi lên là tọa độ pixel hay tọa độ lưới
        sample_x = snake_coords[0][0]
        is_pixel_coords = sample_x > 40 
        
        if is_pixel_coords:
            self.snake = [Point(x, y) for x, y in snake_coords]
            self.food = Point(food_coord[0], food_coord[1])
        else:
            self.snake = [Point(x * BLOCK_SIZE, y * BLOCK_SIZE) for x, y in snake_coords]
            self.food = Point(food_coord[0] * BLOCK_SIZE, food_coord[1] * BLOCK_SIZE)
            
        self.head = self.snake[0]
        
        # Xác định hướng hiện tại dựa vào đầu và cổ rắn
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
        # Check tường
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Check thân rắn
        if pt in self.snake[1:]:
            return True
        return False
    
    # Hàm này cần thiết cho RL (Agent) để tính state chính xác như lúc train
    def is_trap(self, point):
        w_grid = self.w // BLOCK_SIZE
        h_grid = self.h // BLOCK_SIZE
        
        start_x = int(point.x // BLOCK_SIZE)
        start_y = int(point.y // BLOCK_SIZE)
        
        if start_x < 0 or start_x >= w_grid or start_y < 0 or start_y >= h_grid:
            return True 

        # Tạo lưới ảo để check BFS
        grid = np.zeros((w_grid, h_grid), dtype=int)
        for pt in self.snake:
            x_idx = int(pt.x // BLOCK_SIZE)
            y_idx = int(pt.y // BLOCK_SIZE)
            if 0 <= x_idx < w_grid and 0 <= y_idx < h_grid:
                grid[x_idx][y_idx] = 1 
        
        # Đuôi rắn không tính là vật cản vì nó sẽ di chuyển
        tail = self.snake[-1]
        tail_x = int(tail.x // BLOCK_SIZE)
        tail_y = int(tail.y // BLOCK_SIZE)
        grid[tail_x][tail_y] = 0 

        if grid[start_x][start_y] == 1:
            return True
            
        # BFS tìm đường về đuôi
        queue = [(start_x, start_y)]
        visited = set()
        visited.add((start_x, start_y))
        found_tail = False
        
        while queue:
            cx, cy = queue.pop(0)
            if cx == tail_x and cy == tail_y:
                found_tail = True
                break
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w_grid and 0 <= ny < h_grid:
                    if grid[nx][ny] == 0 and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        
        return not found_tail

# ==========================================
# 4. HÀM TÍNH TRẠNG THÁI CHO RL
# ==========================================
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

    # Thêm check trap cho giống logic lúc train
    trap_l = game.is_trap(point_l)
    trap_r = game.is_trap(point_r)
    trap_u = game.is_trap(point_u)
    trap_d = game.is_trap(point_d)
    
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
        game.food.x < game.head.x, # food left
        game.food.x > game.head.x, # food right
        game.food.y < game.head.y, # food up
        game.food.y > game.head.y, # food down
        
        # Trap info (quan trọng cho model đã train)
        trap_straight,
        trap_right,
        trap_left
    ]
    return np.array(state, dtype=int)

# ==========================================
# 5. SETUP SERVER & RL MODEL
# ==========================================
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="templates"), name="static")
templates = Jinja2Templates(directory="templates")

# Load RL Model
rl_model = None
if Linear_QNet:
    try:
        # Input size phải khớp với training (14)
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

class GameInput(BaseModel):
    snake: list
    food: list
    board_size: dict
    algorithm: str

# ==========================================
# 6. ROUTING
# ==========================================
@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/compare")
def compare_page(request: Request):
    return templates.TemplateResponse("compare.html", {"request": request})

@app.post("/predict")
def predict(data: GameInput):
    # Khởi tạo game giả lập từ dữ liệu frontend gửi về
    game = SimulatedGame(data.snake, data.food, data.board_size['width'], data.board_size['height'])
    
    move_vector = [0, 0] # Default move (đứng yên hoặc đi thẳng nếu lỗi)
    next_pt = None

    try:
        # --- GỌI CÁC THUẬT TOÁN TỪ FILE NGOÀI ---
        
        if data.algorithm == "astar":
            # Sử dụng AStarSolver từ solver_A_star.py
            solver = AStarSolver(game)
            next_pt = solver.get_next_move()
            
        elif data.algorithm == "hybrid":
            # Sử dụng HybridSolver từ solver_hybrid.py
            solver = HybridSolver(game)
            next_pt = solver.get_next_move()

        elif data.algorithm == "hamilton":
            # Sử dụng HamiltonSolver từ solver_hamilton.py
            solver = HamiltonSolver(game)
            next_pt = solver.get_next_move()

        elif data.algorithm == "rl" and rl_model:
            # Xử lý Deep Learning
            state = get_game_state(game)
            state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            prediction = rl_model(state0)
            move = torch.argmax(prediction).item()
            
            # Map output của Model (0: thẳng, 1: phải, 2: trái) ra Vector di chuyển
            clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
            idx = clock_wise.index(game.direction)
            
            if move == 0: new_dir = clock_wise[idx] # Thẳng
            elif move == 1: new_dir = clock_wise[(idx + 1) % 4] # Phải
            else: new_dir = clock_wise[(idx - 1) % 4] # Trái

            if new_dir == Direction.RIGHT: move_vector = [1, 0]
            elif new_dir == Direction.LEFT: move_vector = [-1, 0]
            elif new_dir == Direction.UP: move_vector = [0, -1]
            elif new_dir == Direction.DOWN: move_vector = [0, 1]
            
            # RL trả về move_vector luôn, không cần tính từ next_pt
            return {"move": move_vector}

        # --- XỬ LÝ KẾT QUẢ CỦA CÁC THUẬT TOÁN SEARCH (A*, HYBRID, HAMILTON) ---
        if next_pt:
            # Tính vector di chuyển từ tọa độ điểm tiếp theo
            # (1, 0) -> Phải, (-1, 0) -> Trái, (0, -1) -> Lên, (0, 1) -> Xuống
            dx = int((next_pt.x - game.head.x) / BLOCK_SIZE)
            dy = int((next_pt.y - game.head.y) / BLOCK_SIZE)
            move_vector = [dx, dy]
        else:
            # Nếu thuật toán không tìm được đường (bị kẹt), đi theo quán tính
            # Để tránh crash web
            if game.direction == Direction.RIGHT: move_vector = [1, 0]
            elif game.direction == Direction.LEFT: move_vector = [-1, 0]
            elif game.direction == Direction.UP: move_vector = [0, -1]
            elif game.direction == Direction.DOWN: move_vector = [0, 1]

    except Exception as e:
        print(f"❌ Lỗi Xử Lý Thuật Toán {data.algorithm}: {e}")
        import traceback
        traceback.print_exc()

    return {"move": move_vector}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)