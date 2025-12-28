import torch
import random
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

# --- HACK MODULE ĐỂ CÁC FILE SOLVER KHÔNG BỊ LỖI ---
# Tạo module giả 'game' và 'snake_game' để solver_hybrid.py có thể import
game_module = types.ModuleType("game")
game_module.Direction = Direction
game_module.Point = Point
game_module.BLOCK_SIZE = BLOCK_SIZE
sys.modules["game"] = game_module
sys.modules["snake_game"] = game_module 

# Import Model RL
try:
    from model import Linear_QNet
except ImportError:
    print("⚠️ Warning: model.py not found")
    Linear_QNet = None

# Import Solvers
try:
    from solver_hybrid import HybridSolver
    from solver_hamilton import HamiltonSolver
    from solver_A_star import AStarSolver
except ImportError as e:
    print(f"⚠️ Lỗi Import Solver: {e}. Đảm bảo các file solver nằm cùng thư mục.")
    # Fallback để code không crash ngay lập tức
    class HybridSolver: pass
    class HamiltonSolver: pass
    class AStarSolver: pass

# ==========================================
# 2. CLASS GIẢ LẬP (CORE LOGIC)
# ==========================================
class SimulatedGame:
    def __init__(self, snake_coords, food_coord, width_cells, height_cells):
        self.block_size = BLOCK_SIZE
        # Kích thước Pixel (ví dụ: 640x480)
        self.w = width_cells * BLOCK_SIZE
        self.h = height_cells * BLOCK_SIZE
        # Kích thước Grid (ví dụ: 32x24)
        self.cols = width_cells
        self.rows = height_cells
        
        # --- CHUẨN HÓA TỌA ĐỘ (GRID -> PIXEL) ---
        # Frontend gửi Grid Index (ví dụ: 15, 12). Ta nhân 20 để thành Pixel (300, 240)
        # để khớp logic với các thuật toán đã viết cho Pygame.
        self.snake = [Point(pt[0] * BLOCK_SIZE, pt[1] * BLOCK_SIZE) for pt in snake_coords]
        self.food = Point(food_coord[0] * BLOCK_SIZE, food_coord[1] * BLOCK_SIZE)
            
        self.head = self.snake[0]
        self.score = len(self.snake) - 3
        
        # Xác định hướng hiện tại dựa vào đầu và cổ rắn
        if len(self.snake) > 1:
            neck = self.snake[1]
            if self.head.x > neck.x: self.direction = Direction.RIGHT
            elif self.head.x < neck.x: self.direction = Direction.LEFT
            elif self.head.y < neck.y: self.direction = Direction.UP   # Pygame: y giảm là lên
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

    # --- HÀM IS_TRAP (CỰC KỲ QUAN TRỌNG CHO RL) ---
    def is_trap(self, point):
        """
        Kiểm tra xem đi vào điểm 'point' có bị kẹt (không về được đuôi) hay không.
        True = Nguy hiểm (Trap/Ngõ cụt). False = An toàn.
        """
        # 1. Nếu đâm tường hoặc đâm thân ngay lập tức -> Trap
        if self.is_collision(point): 
            return True
            
        # 2. Chuẩn bị dữ liệu cho BFS (Chuyển về Grid Index cho nhẹ)
        start_node = (int(point.x // BLOCK_SIZE), int(point.y // BLOCK_SIZE))
        
        tail = self.snake[-1]
        tail_node = (int(tail.x // BLOCK_SIZE), int(tail.y // BLOCK_SIZE))

        # Tạo tập vật cản (Set lookup O(1))
        obstacles = set()
        for pt in self.snake:
            ox = int(pt.x // BLOCK_SIZE)
            oy = int(pt.y // BLOCK_SIZE)
            obstacles.add((ox, oy))
        
        # QUAN TRỌNG: Khi rắn di chuyển, đuôi sẽ đi chỗ khác -> Đuôi không phải vật cản
        if tail_node in obstacles:
            obstacles.remove(tail_node)
            
        # Nếu điểm xét trùng vật cản -> Trap
        if start_node in obstacles: 
            return True
        
        # Nếu điểm xét chính là đuôi -> An toàn tuyệt đối
        if start_node == tail_node:
            return False

        # 3. BFS tìm đường từ point -> tail
        queue = deque([start_node])
        visited = set([start_node])
        
        while queue:
            cx, cy = queue.popleft()
            
            # Nếu tìm thấy đuôi -> Có đường thoát -> Không phải Trap
            if (cx, cy) == tail_node:
                return False 

            # Duyệt 4 hướng
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                
                # Check trong biên
                if 0 <= nx < self.cols and 0 <= ny < self.rows:
                    if (nx, ny) not in obstacles and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        
        # Hết queue mà không thấy đuôi -> Kẹt -> Là Trap
        return True 

# ==========================================
# 3. HÀM TÍNH STATE VECTOR (14 GIÁ TRỊ)
# ==========================================
def get_game_state(game):
    head = game.head
    
    # Các điểm lân cận (Pixel)
    point_l = Point(head.x - BLOCK_SIZE, head.y)
    point_r = Point(head.x + BLOCK_SIZE, head.y)
    point_u = Point(head.x, head.y - BLOCK_SIZE)
    point_d = Point(head.x, head.y + BLOCK_SIZE)
    
    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN

    # --- TÍNH TOÁN TRAP (Logic này quyết định độ khôn của RL) ---
    trap_l = game.is_trap(point_l)
    trap_r = game.is_trap(point_r)
    trap_u = game.is_trap(point_u)
    trap_d = game.is_trap(point_d)
    
    # Map Trap tương đối theo hướng di chuyển (Straight, Right, Left)
    # Ví dụ: Đang đi lên (UP), thì Straight=UP, Right=RIGHT, Left=LEFT
    
    # Straight
    trap_straight = (dir_r and trap_r) or (dir_l and trap_l) or (dir_u and trap_u) or (dir_d and trap_d)
    
    # Right
    trap_right = (dir_u and trap_r) or (dir_d and trap_l) or (dir_l and trap_u) or (dir_r and trap_d)
    
    # Left
    trap_left = (dir_d and trap_r) or (dir_u and trap_l) or (dir_r and trap_u) or (dir_l and trap_d)

    state = [
        # [0-3] Danger collision (Tường/Thân ngay cạnh) - Relative
        (dir_r and game.is_collision(point_r)) or 
        (dir_l and game.is_collision(point_l)) or 
        (dir_u and game.is_collision(point_u)) or 
        (dir_d and game.is_collision(point_d)), # Danger Straight

        (dir_u and game.is_collision(point_r)) or 
        (dir_d and game.is_collision(point_l)) or 
        (dir_l and game.is_collision(point_u)) or 
        (dir_r and game.is_collision(point_d)), # Danger Right

        (dir_d and game.is_collision(point_r)) or 
        (dir_u and game.is_collision(point_l)) or 
        (dir_r and game.is_collision(point_u)) or 
        (dir_l and game.is_collision(point_d)), # Danger Left
        
        # [4-7] Move direction
        dir_l, dir_r, dir_u, dir_d,
        
        # [8-11] Food location
        game.food.x < game.head.x, # food left
        game.food.x > game.head.x, # food right
        game.food.y < game.head.y, # food up
        game.food.y > game.head.y, # food down
        
        # [12-14] Trap info (QUAN TRỌNG)
        int(trap_straight),
        int(trap_right),
        int(trap_left)
    ]
    return np.array(state, dtype=int)

# ==========================================
# 4. SERVER SETUP
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
try:
    if Linear_QNet:
        # Input size = 14 (Khớp với get_game_state)
        rl_model = Linear_QNet(14, 256, 3)
        model_path = os.path.join(BASE_DIR, 'model', 'model.pth')
        if os.path.exists(model_path):
            rl_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            rl_model.eval()
            print("✅ RL Model Loaded Successfully")
        else:
            print("⚠️ Warning: model.pth not found in ./model/")
except Exception as e:
    print(f"⚠️ Error loading RL model: {e}")

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
    # Khởi tạo game giả lập
    game = SimulatedGame(data.snake, data.food, data.board_size['width'], data.board_size['height'])
    move_vector = [0, 0]

    try:
        # ----------------------------------
        # TRƯỜNG HỢP 1: THUẬT TOÁN DEEP RL
        # ----------------------------------
        if data.algorithm == "rl" and rl_model:
            state = get_game_state(game)
            state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            
            # Dự đoán nước đi
            with torch.no_grad():
                prediction = rl_model(state0)
            
            move = torch.argmax(prediction).item()
            
            # Map: [0: Straight, 1: Right, 2: Left] -> Direction
            clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
            try: 
                idx = clock_wise.index(game.direction)
            except: 
                idx = 0 # Default nếu lỗi
            
            if move == 0: new_dir = clock_wise[idx] 
            elif move == 1: new_dir = clock_wise[(idx + 1) % 4] 
            else: new_dir = clock_wise[(idx - 1) % 4] 
            
            # Chuyển Direction thành Vector [dx, dy]
            if new_dir == Direction.RIGHT: move_vector = [1, 0]
            elif new_dir == Direction.LEFT: move_vector = [-1, 0]
            elif new_dir == Direction.UP: move_vector = [0, -1]
            elif new_dir == Direction.DOWN: move_vector = [0, 1]

        # ----------------------------------
        # TRƯỜNG HỢP 2: CÁC THUẬT TOÁN SEARCH
        # ----------------------------------
        else:
            solver = None
            if data.algorithm == "astar":
                solver = AStarSolver(game)
            elif data.algorithm == "hybrid":
                solver = HybridSolver(game)
            elif data.algorithm == "hamilton":
                solver = HamiltonSolver(game)

            next_pt = None
            if solver:
                next_pt = solver.get_next_move()

            if next_pt:
                # Tính vector di chuyển
                dx = int((next_pt.x - game.head.x) / BLOCK_SIZE)
                dy = int((next_pt.y - game.head.y) / BLOCK_SIZE)
                move_vector = [dx, dy]
            else:
                # FALLBACK: Nếu thuật toán bó tay (trả về None)
                # Đi thẳng để hy vọng sống sót thêm 1 turn
                if game.direction == Direction.RIGHT: move_vector = [1, 0]
                elif game.direction == Direction.LEFT: move_vector = [-1, 0]
                elif game.direction == Direction.UP: move_vector = [0, -1]
                elif game.direction == Direction.DOWN: move_vector = [0, 1]

    except Exception as e:
        print(f"❌ Error in predict: {e}")
        traceback.print_exc()

    return {"move": move_vector}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)