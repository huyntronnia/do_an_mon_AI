from utils import BaseSolver
import random

class AStarSolver(BaseSolver):
    def get_next_move(self):
        start = self.game.head
        food = self.game.food
        snake = self.game.snake
        
        # 1. Xác định vật cản
        # Lấy toàn bộ thân rắn
        obstacles = set(snake)
        
        # QUAN TRỌNG: Phải bỏ cái ĐẦU ra khỏi vật cản, nếu không A* sẽ báo không có đường đi ngay lập tức
        if start in obstacles:
            obstacles.remove(start)
            
        # Bỏ cái ĐUÔI ra (vì rắn di chuyển thì đuôi cũng chạy đi, ô đó thành đường đi được)
        tail = snake[-1]
        if tail in obstacles:
            obstacles.remove(tail)
        
        # 2. Tìm đường ngắn nhất đến mồi
        path = self.a_star_path(start, food, obstacles)
        
        # 3. Di chuyển
        if path and len(path) > 1:
            return path[1]
        
        # 4. Fallback (Cứu cánh): Nếu bị kẹt (không tìm thấy đường A*)
        # Đi đại vào một ô trống bất kỳ để mong sống sót thêm 1 lượt
        neighbors = self.get_neighbors(start)
        valid_moves = [n for n in neighbors if n not in obstacles]
        
        if valid_moves:
            # Ưu tiên đi vào ô nào có khoảng trống rộng nhất (Flood Fill đơn giản)
            # Để tránh chui vào ngõ cụt bé tí
            best_move = max(valid_moves, key=lambda m: self.bfs_flood_fill(m, obstacles))
            return best_move
                
        return None # Chết chắc