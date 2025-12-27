from snake_game import SnakeGameAI, Direction
import pygame

# --- IMPORT CÁC THUẬT TOÁN ---
from solver_A_star import AStarSolver
from solver_hamilton import HamiltonSolver
from solver_hybrid import HybridSolver

# --- CẤU HÌNH THUẬT TOÁN ---
# Chọn: 'ASTAR' | 'HAMILTON' | 'HYBRID'
ALGORITHM = 'HAMILTON' 
# ---------------------------

def get_direction_from_points(from_pt, to_pt):
    if to_pt.x > from_pt.x: return Direction.RIGHT
    if to_pt.x < from_pt.x: return Direction.LEFT
    if to_pt.y > from_pt.y: return Direction.DOWN
    if to_pt.y < from_pt.y: return Direction.UP
    return Direction.RIGHT

if __name__ == '__main__':
    game = SnakeGameAI()
    
    # Khởi tạo Solver dựa trên cấu hình
    if ALGORITHM == 'ASTAR':
        solver = AStarSolver(game)
    elif ALGORITHM == 'HAMILTON':
        solver = HamiltonSolver(game)
    else:
        solver = HybridSolver(game)

    pygame.display.set_caption(f'Snake AI - Mode: {ALGORITHM}')
    
    while True:
        # 1. Lấy nước đi tiếp theo
        next_point = solver.get_next_move()
        
        # 2. (Tùy chọn) Vẽ đường đi để Debug
        # Chỉ vẽ được nếu Solver có hàm tìm đường (Hamilton ko cần vẽ)
        path_to_draw = []
        if ALGORITHM != 'HAMILTON':
            obstacles = set(game.snake[:-1])
            # Thử tìm đường đến mồi để vẽ
            temp_path = solver.a_star_path(game.head, game.food, obstacles)
            if temp_path:
                path_to_draw = temp_path
            # Nếu Hybrid đang tìm đuôi thì vẽ đường về đuôi
            elif ALGORITHM == 'HYBRID':
                path_to_tail = solver.a_star_path(game.head, game.snake[-1], obstacles)
                if path_to_tail: path_to_draw = path_to_tail
        
        game.set_path(path_to_draw)

        # 3. Điều khiển game
        if next_point:
            action = get_direction_from_points(game.head, next_point)
        else:
            action = game.direction 
            
        game_over, score = game.play_step(action)

        if game_over:
            print(f"GAME OVER ({ALGORITHM})! Score: {score}")
            pygame.time.wait(2000)
            game.reset()
            # Nếu muốn đổi thuật toán sau khi chết, bạn có thể code thêm logic ở đây
            
    pygame.quit()