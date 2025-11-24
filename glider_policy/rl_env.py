# glider_rnn_env.py
import gym
import numpy as np
from gym import spaces

from policy_trainer import (   # 把这里换成你 RNN 代码所在的文件名
    load_trajectory,
    compute_dt,
    find_learn_segment,
    simulate_step,
)


class GliderRNNEnv(gym.Env):
    """
    用于对 RNN policy 进行 RL 微调的滑翔机环境。

    - 使用与你 RNN policy 一致的状态定义和动力学 simulate_step
    - observation: (n_hist, 9)
        每一行: [v, λ_rel, φ_rel, r_rel, psi, gamma, goal_rel(3)]
    - action: [Δv, Δpsi, Δgamma] ∈ [-1,0]×[-1,1]×[-1,1]
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        csv_path,
        n_hist=10,
        max_steps=200,
        start_from_learn_segment=True,
        arrival_threshold_m=1_000.0,
        distance_scale_m=50_000.0,
        imitation_weight=0.0,
    ):
        super().__init__()

        # -------- 1. 读入轨迹数据（与你 RNN 代码同一套 CSV） --------
        states_abs, h, t = load_trajectory(csv_path)   # (T, 6), (T,), (T,)
        self.states_abs = states_abs.astype(np.float32)
        self.h = h.astype(np.float32)
        self.t = t.astype(np.float32)
        self.T = len(self.states_abs)

        if self.T < n_hist + 2:
            raise ValueError(f"轨迹 {csv_path} 太短，无法构造 n_hist={n_hist} 的历史窗口。")

        self.dt = compute_dt(self.t)

        # 参考原点（绝对坐标的第一帧）
        self.lam0, self.phi0, self.r0 = self.states_abs[0, 1:4]

        # 终点（绝对 & 相对）
        self.goal_abs = self.states_abs[-1, 1:4].copy()
        self.goal_rel = np.array(
            [self.goal_abs[0] - self.lam0,
             self.goal_abs[1] - self.phi0,
             self.goal_abs[2] - self.r0],
            dtype=np.float32
        )

        # 相对坐标状态
        self.states_rel = self.states_abs.copy()
        self.states_rel[:, 1] -= self.lam0
        self.states_rel[:, 2] -= self.phi0
        self.states_rel[:, 3] -= self.r0

        # “学习段”起点（和你 RNN 训练里一致）
        learn_start = find_learn_segment(self.states_abs[:, 0], self.h)
        if learn_start is None:
            learn_start = 0
        self.start_index = learn_start if start_from_learn_segment else 0

        # -------- 2. 环境参数 --------
        self.n_hist = n_hist
        self.max_steps = max_steps
        self.arrival_threshold_m = float(arrival_threshold_m)
        self.distance_scale_m = float(distance_scale_m)
        self.imitation_weight = float(imitation_weight)

        self.current_idx = None
        self.current_step = None
        self.current_state_rel = None
        self.current_state_abs = None
        self.history_states = None

        # -------- 3. action space：跟 RNN 一致 --------
        action_low = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
        action_high = np.array([0.0, 1.0, 1.0], dtype=np.float32)
        self.action_space = spaces.Box(
            low=action_low, high=action_high, shape=(3,), dtype=np.float32
        )

        # -------- 4. observation space: (n_hist, 9) --------
        # SB3 会自动 flatten 成 1D，这里用一个 大 Box 即可
        obs_low = np.full((n_hist, 9), -np.inf, dtype=np.float32)
        obs_high = np.full((n_hist, 9), np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )

    # =========================================
    # reset
    # =========================================
    def reset(self):
        self.current_idx = int(self.start_index)
        self.current_step = 0

        # 初始状态 = 真实轨迹对应帧
        self.current_state_abs = self.states_abs[self.current_idx].copy()
        self.current_state_rel = self.states_rel[self.current_idx].copy()

        # 历史窗口：往前补，不足用当前状态填
        self.history_states = []
        for k in range(self.n_hist):
            idx = self.current_idx - (self.n_hist - 1 - k)
            if idx >= 0:
                self.history_states.append(self.states_rel[idx].copy())
            else:
                self.history_states.append(self.current_state_rel.copy())

        obs = self._build_observation()
        return obs

    # =========================================
    # step
    # =========================================
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        prev_distance_m = self._distance_to_goal_m(self.current_state_rel)

        # 用与你 test_policy 一致的动力学
        next_rel, next_abs = simulate_step(
            state_rel=self.current_state_rel,
            state_abs=self.current_state_abs,
            action=action,
            dt=self.dt,
            lam0=self.lam0,
            phi0=self.phi0,
            r0=self.r0,
        )

        self.current_state_rel = next_rel
        self.current_state_abs = next_abs

        # 更新历史
        self.history_states.append(next_rel.copy())
        if len(self.history_states) > self.n_hist:
            self.history_states.pop(0)

        self.current_step += 1
        self.current_idx = min(self.current_idx + 1, self.T - 1)

        obs = self._build_observation()

        distance_m = self._distance_to_goal_m(self.current_state_rel)
        reward, info = self._compute_reward(
            prev_distance_m=prev_distance_m,
            distance_m=distance_m,
            action=action
        )

        done = (distance_m < self.arrival_threshold_m) or (self.current_step >= self.max_steps)

        return obs, float(reward), bool(done), info

    # =========================================
    # 观测构造
    # =========================================
    def _build_observation(self):
        hist_arr = np.stack(self.history_states, axis=0)  # (n_hist, 6)
        goal_tile = np.repeat(self.goal_rel[None, :], self.n_hist, axis=0)  # (n_hist, 3)
        seq_input = np.concatenate([hist_arr, goal_tile], axis=-1)          # (n_hist, 9)
        return seq_input.astype(np.float32)

    # =========================================
    # 距离终点
    # =========================================
    def _distance_to_goal_m(self, state_rel):
        R_earth = float(self.r0)
        cos_phi0 = np.cos(float(self.phi0))

        delta = self.goal_rel - state_rel[1:4]

        d_lambda_m = R_earth * cos_phi0 * delta[0]
        d_phi_m = R_earth * delta[1]
        d_r_m = delta[2]

        distance_m = np.sqrt(d_lambda_m**2 + d_phi_m**2 + d_r_m**2)
        return float(distance_m)

    # =========================================
    # 奖励函数：靠近终点 + 动作惩罚 (+ 可选模仿)
    # =========================================
    def _compute_reward(self, prev_distance_m, distance_m, action):
        base_reward = np.exp(-distance_m / self.distance_scale_m)
        progress_reward = (prev_distance_m - distance_m) / self.distance_scale_m

        end_bonus = 0.0
        if distance_m < 5_000.0:
            end_bonus = 5.0
        if distance_m < 2_000.0:
            end_bonus = 10.0

        action_norm = float(np.linalg.norm(action))
        action_penalty = 0.1 * (action_norm ** 2)

        imitation_penalty = 0.0
        if self.imitation_weight > 0.0:
            ref_rel = self.states_rel[self.current_idx]
            delta_ref = self.current_state_rel[1:4] - ref_rel[1:4]
            pos_error_m = self._rel_pos_to_m(delta_ref)
            imitation_penalty = self.imitation_weight * (pos_error_m ** 2) * 1e-8

        reward = (
            base_reward
            + progress_reward
            + end_bonus
            - action_penalty
            - imitation_penalty
        )

        info = {
            "distance_to_goal_m": float(distance_m),
            "base_reward": float(base_reward),
            "progress_reward": float(progress_reward),
            "end_bonus": float(end_bonus),
            "action_penalty": float(action_penalty),
            "imitation_penalty": float(imitation_penalty),
            "reward": float(reward),
        }
        return reward, info

    def _rel_pos_to_m(self, delta_rel):
        R_earth = float(self.r0)
        cos_phi0 = np.cos(float(self.phi0))

        d_lambda_m = R_earth * cos_phi0 * delta_rel[0]
        d_phi_m = R_earth * delta_rel[1]
        d_r_m = delta_rel[2]

        return float(np.sqrt(d_lambda_m**2 + d_phi_m**2 + d_r_m**2))

    def render(self, mode="human"):
        return None
