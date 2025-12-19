import torch
import numpy as np
from .base import BaseTask

class RoverTrajectoryTask(BaseTask):
    """
    60维 Rover 轨迹规划任务。
    目标：规划一条由 30 个点 (x, y) 组成的路径，
    使其尽可能短（平滑）且避开障碍物。
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.n_points = 30
        self._dim = self.n_points * 2 
        
        # 定义环境
        self.start = np.array([0.0, 0.0])
        self.goal = np.array([1.0, 1.0])
        
        # 障碍物定义: [x, y, radius]
        self.obstacles = np.array([
            [0.5, 0.5, 0.2],
            [0.2, 0.8, 0.15],
            [0.8, 0.2, 0.15]
        ])

    @property
    def dim(self) -> int:
        return self._dim

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 60) in [0, 1]
        batch_size = x.size(0)
        rewards = []
        x_np = x.detach().cpu().numpy()
        
        for i in range(batch_size):
            # 1. 重构轨迹: (30, 2)
            # 因为 BO 输出就在 [0, 1]，正好对应地形坐标，无需额外映射
            traj = x_np[i].reshape(self.n_points, 2)
            
            # 拼接起点和终点 -> (32, 2)
            full_traj = np.vstack([self.start, traj, self.goal])
            
            # 2. 计算代价
            # A. 路径长度代价 (平滑度)
            segment_lengths = np.linalg.norm(np.diff(full_traj, axis=0), axis=1)
            cost_length = np.sum(segment_lengths)
            
            # B. 障碍物碰撞代价
            cost_collision = 0.0
            for obs in self.obstacles:
                # 计算所有轨迹点到圆心的距离
                # 广播计算: (32, 2) - (2,) -> (32, 2) -> norm -> (32,)
                dists = np.linalg.norm(full_traj - obs[:2], axis=1)
                
                # 硬碰撞惩罚
                collision_mask = dists < obs[2]
                if np.any(collision_mask):
                    cost_collision += 100.0 * np.sum(collision_mask)
                
                # 软距离惩罚 (排斥场)
                # 当距离接近半径时，代价指数上升
                margin = dists - obs[2]
                # 只对靠近障碍物的点计算斥力
                influence_mask = margin < 0.1 
                if np.any(influence_mask):
                    # exp(-alpha * distance)
                    cost_collision += np.sum(np.exp(-20.0 * margin[influence_mask]))

            # 3. 目标是最大化 Reward，所以取负 Cost
            # 为了数值稳定性，通常归一化一下或者限制范围
            total_cost = cost_length + cost_collision
            rewards.append(-total_cost)
            
        return torch.tensor(rewards, device=self.device, dtype=self.dtype).unsqueeze(-1)
