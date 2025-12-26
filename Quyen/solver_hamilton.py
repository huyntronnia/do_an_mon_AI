from utils import BaseSolver
from snake_game import Point, BLOCK_SIZE

class HamiltonSolver(BaseSolver):
    def __init__(self, game):
        super().__init__(game)
        # Chỉ cần xây dựng chu trình 1 lần duy nhất
        self.hamiltonian_path = self._build_hamiltonian_cycle()

    def _build_hamiltonian_cycle(self):
        """
        Tạo đường đi Zig-Zag khép kín phủ đầy bản đồ.
        Yêu cầu: Số cột (W/BLOCK) phải là số CHẴN để chu trình hoàn hảo.
        """
        path_map = {}
        cols = self.game.w // BLOCK_SIZE
        rows = self.game.h // BLOCK_SIZE

        # 1. Xây dựng đường Zig-Zag (Chừa hàng trên cùng y=0 làm đường về)
        for x in range(0, self.game.w, BLOCK_SIZE):
            col_idx = x // BLOCK_SIZE
            
            # Cột CHẴN (0, 2, 4...): Đi XUỐNG
            if col_idx % 2 == 0:
                # Đi từ y=BLOCK_SIZE xuống đáy
                for y in range(BLOCK_SIZE, self.game.h - BLOCK_SIZE, BLOCK_SIZE):
                    path_map[Point(x, y)] = Point(x, y + BLOCK_SIZE)
                
                # Tại đáy: Sang PHẢI
                bottom_y = self.game.h - BLOCK_SIZE
                path_map[Point(x, bottom_y)] = Point(x + BLOCK_SIZE, bottom_y)

            # Cột LẺ (1, 3, 5...): Đi LÊN
            else:
                # Đi từ đáy lên y=BLOCK_SIZE
                for y in range(self.game.h - BLOCK_SIZE, BLOCK_SIZE, -BLOCK_SIZE):
                    path_map[Point(x, y)] = Point(x, y - BLOCK_SIZE)
                
                # Tại đỉnh cột lẻ (y=BLOCK_SIZE):
                top_inner_y = BLOCK_SIZE
                
                if col_idx < cols - 1:
                    # Nếu chưa phải cột cuối cùng -> Sang PHẢI (để sang cột chẵn tiếp theo)
                    path_map[Point(x, top_inner_y)] = Point(x + BLOCK_SIZE, top_inner_y)
                else:
                    # Nếu là cột cuối cùng -> Đi LÊN (vào hàng về y=0)
                    path_map[Point(x, top_inner_y)] = Point(x, 0)

        # 2. Xây dựng Hàng Về (Highway row y=0): Đi một mạch từ PHẢI về TRÁI
        for x in range(self.game.w - BLOCK_SIZE, 0, -BLOCK_SIZE):
            path_map[Point(x, 0)] = Point(x - BLOCK_SIZE, 0)
        
        # 3. Khép vòng: Từ (0,0) xuống (0, BLOCK_SIZE)
        path_map[Point(0, 0)] = Point(0, BLOCK_SIZE)

        return path_map

    def get_next_move(self):
        # --- LOGIC BẤT TỬ ---
        # Không suy nghĩ, không tính toán, không đi tắt.
        # Chỉ tra bảng và đi theo chỉ dẫn.
        
        current_pos = self.game.head
        
        if current_pos in self.hamiltonian_path:
            return self.hamiltonian_path[current_pos]
        else:
            # Trường hợp hiếm: Rắn bị spawn ở vị trí lệch chu trình (thường không xảy ra nếu map chuẩn)
            # Đi đại hướng phải để tìm đường
            return Point(current_pos.x + BLOCK_SIZE, current_pos.y)