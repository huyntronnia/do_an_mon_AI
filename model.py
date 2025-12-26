import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# --- BƯỚC 3: THIẾT KẾ BỘ NÃO (Deep Q-Network Model) ---
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Lớp Input -> Hidden (11 -> 256)
        self.linear1 = nn.Linear(input_size, hidden_size)
        # Lớp Hidden -> Output (256 -> 3)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Hàm kích hoạt ReLU cho lớp ẩn
        x = F.relu(self.linear1(x))
        # Lớp đầu ra không cần hàm kích hoạt (raw values - Q values)
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        # Lưu model để dùng lại sau này
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss() # Hàm mất mát Mean Squared Error

    def train_step(self, state, action, reward, next_state, done):
        # Chuyển đổi dữ liệu sang Tensor
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # Xử lý nếu chỉ có 1 mẫu (khi train short memory)
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1. Dự đoán Q-values hiện tại
        pred = self.model(state)

        # 2. Tính Q-new theo công thức Bellman: Q_new = r + y * max(next_predicted_Q)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            # Chỉ cập nhật giá trị Q cho hành động đã thực hiện
            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 3. Backpropagation (Lan truyền ngược)
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()