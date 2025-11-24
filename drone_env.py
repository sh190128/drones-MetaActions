import gym
import numpy as np
from gym import spaces


class DroneEnv(gym.Env):
    """
    自定义无人机轨迹跟踪环境

    X: shape (N, 6)
       每一行: [v, lambda, phi, r, psi, gamma]
    y: shape (N, 3) 或 (N-1, 3)
       每一行: [lambda_next, phi_next, r_next]
       约定: y[t] 是从 X[t] 这个状态出发，下一时刻的位置
    """
    metadata = {"render.modes": []}

    def __init__(self, X, y, dt, max_steps=100, test=False, n_obs=5):
        super(DroneEnv, self).__init__()

        self.dt = dt
        self.test = test
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.n_obs = n_obs  # 历史观测窗口大小

        # 轨迹长度取 X 和 y 的最小长度，防止越界
        self.trajectory_length = min(len(self.X), len(self.y))

        self.max_steps = max_steps
        self.current_step = 0          # 在当前 episode 中走了多少步
        self.current_episode = 0       # 当前 episode 的起始索引（在整条轨迹中的起点）

        # 以第一帧的经度、纬度、地心距作为参考原点，后面全部用“相对坐标”
        self.longitude_start = self.X[0][1]  # lambda
        self.latitude_start = self.X[0][2]   # phi
        self.r_start = self.X[0][3]

        # --------------------
        # 定义 action space
        # --------------------
        # 动作: [Δv, Δpsi, Δgamma]，每一维 ∈ [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # --------------------
        # 定义 observation space
        # --------------------
        # 单个状态: [v, lambda, phi, r, psi, gamma]
        state_low = np.array([
            0.0,          # v
            -np.pi,       # lambda
            -np.pi / 2,   # phi
            6_370_000.0,  # r
            0.0,          # psi
            -np.pi        # gamma
        ], dtype=np.float32)

        state_high = np.array([
            10_000.0,     # v
            np.pi,        # lambda
            np.pi / 2,    # phi
            6_500_000.0,  # r
            2 * np.pi,    # psi
            np.pi         # gamma
        ], dtype=np.float32)

        # 历史 n_obs 个状态
        history_low = np.tile(state_low, n_obs)
        history_high = np.tile(state_high, n_obs)

        # 终点位置 [lambda, phi, r]
        endpoint_low = np.array([-np.pi, -np.pi / 2, 6_370_000.0], dtype=np.float32)
        endpoint_high = np.array([np.pi, np.pi / 2, 6_500_000.0], dtype=np.float32)

        # 剩余步数比例 [0, 1]
        steps_low = np.array([0.0], dtype=np.float32)
        steps_high = np.array([1.0], dtype=np.float32)

        obs_low = np.concatenate([history_low, endpoint_low, steps_low])
        obs_high = np.concatenate([history_high, endpoint_high, steps_high])

        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )

        # 当前状态（相对坐标）、终点信息
        self.state = None               # 当前状态 (相对坐标系)
        self.target_pos = None          # 期望下一步位置 (相对坐标)
        self.trajectory_end = None      # 轨迹终点 (绝对坐标)

        # 历史状态窗口
        self.history_states = []

    # -------------------------------------------------
    # reset: 初始化一个新的 episode
    # -------------------------------------------------
    def reset(self):
        # 训练模式：随机截取一段子轨迹作为一个 episode
        if not self.test:
            # 预留 max_steps 的长度，防止后续索引越界
            max_start = max(self.trajectory_length - self.max_steps - 1, 0)
            if max_start > 0:
                self.current_episode = np.random.randint(0, max_start)
            else:
                self.current_episode = 0
        else:
            # 测试模式下可以固定从 0 开始，或者外部提前设定 self.current_episode
            self.current_episode = min(self.current_episode, self.trajectory_length - 2)

        # 当前 step 重置
        self.current_step = 0

        # 当前状态（绝对坐标来自 X）
        abs_state = self.X[self.current_episode].copy()

        # 把 state 转为相对坐标（相对第一帧）
        self.state = abs_state.copy()
        self.state[1] -= self.longitude_start
        self.state[2] -= self.latitude_start
        self.state[3] -= self.r_start

        # 终点的绝对位置（这里用 y 的最后一个位置）
        self.trajectory_end = self.y[self.trajectory_length - 1, 0:3].copy()

        # 终点转为相对坐标
        trajectory_end_relative = np.array([
            self.trajectory_end[0] - self.longitude_start,
            self.trajectory_end[1] - self.latitude_start,
            self.trajectory_end[2] - self.r_start
        ], dtype=np.float32)

        # 初始化历史状态窗口
        self.history_states = []

        # 构造 [t-n+1, ..., t-1, t] 历史窗口，不足用当前状态填充
        while len(self.history_states) < self.n_obs:
            idx = self.current_episode - (self.n_obs - len(self.history_states))
            if idx >= 0:
                prev_abs_state = self.X[idx].copy()
                prev_abs_state[1] -= self.longitude_start
                prev_abs_state[2] -= self.latitude_start
                prev_abs_state[3] -= self.r_start
                self.history_states.append(prev_abs_state)
            else:
                self.history_states.append(self.state.copy())

        # 初始观察量
        steps_remaining_ratio = 1.0  # 初始时还没走步
        flat_history = np.concatenate(self.history_states)
        obs = np.concatenate([flat_history, trajectory_end_relative, [steps_remaining_ratio]]).astype(np.float32)

        return obs

    # -------------------------------------------------
    # step: 环境一步演化
    # -------------------------------------------------
    def step(self, action):
        # 安全处理 action
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # 当前在轨迹中的索引
        idx = self.current_episode + self.current_step
        # 防止越界：如果已经到头，直接结束
        if idx >= self.trajectory_length - 1:
            return self._terminal_observation(), 0.0, True, {}

        # 从 y 中取“下一步的真实位置”，并转相对坐标（只用于 info/评估）
        target_abs = self.y[idx].copy()  # [lambda_next, phi_next, r_next]
        self.target_pos = target_abs - np.array(
            [self.longitude_start, self.latitude_start, self.r_start],
            dtype=np.float32
        )

        # ------------------------
        # 1. 根据动作更新状态
        # ------------------------
        V = self.state[0]
        lambda_rel = self.state[1]
        phi_rel = self.state[2]
        r_rel = self.state[3]
        psi = self.state[4]
        gamma = self.state[5]

        V_change = action[0] * 10.0      # 速度调整范围 ±10 m/s
        psi_change = action[1] * 0.01    # 航向角调整范围 ±0.01 rad
        gamma_change = action[2] * 0.01  # 弹道倾角调整范围 ±0.01 rad

        new_V = np.clip(V + V_change, 0.0, 10_000.0)
        new_psi = psi + psi_change
        new_gamma = gamma + gamma_change

        horizontal_V = new_V * np.cos(new_gamma)
        vertical_V = new_V * np.sin(new_gamma)

        abs_phi = phi_rel + self.latitude_start
        abs_r = r_rel + self.r_start

        dlambda = horizontal_V * self.dt * np.sin(new_psi) / (abs_r * np.cos(abs_phi))
        dphi = horizontal_V * self.dt * np.cos(new_psi) / abs_r
        dr = vertical_V * self.dt

        new_state = np.array([
            new_V,
            lambda_rel + dlambda,
            phi_rel + dphi,
            r_rel + dr,
            new_psi,
            new_gamma
        ], dtype=np.float32)

        # 更新历史窗口
        self.history_states.append(new_state.copy())
        if len(self.history_states) > self.n_obs:
            self.history_states.pop(0)

        self.state = new_state

        # ------------------------
        # 2. 计算奖励（state 跟踪 + 终点形状）
        # ------------------------
        predicted_state = new_state  # [v, λ_rel, φ_rel, r_rel, ψ, γ]

        # 参考下一状态：来自 X[idx+1]
        ref_next_abs = self.X[idx + 1].copy()
        ref_next_rel = ref_next_abs.copy()
        ref_next_rel[1] -= self.longitude_start
        ref_next_rel[2] -= self.latitude_start
        ref_next_rel[3] -= self.r_start

        # 终点相对位置
        trajectory_end_relative = np.array([
            self.trajectory_end[0] - self.longitude_start,
            self.trajectory_end[1] - self.latitude_start,
            self.trajectory_end[2] - self.r_start
        ], dtype=np.float32)

        # 距离终点（米）
        R_earth = float(self.r_start)
        cos_phi0 = np.cos(float(self.latitude_start))

        delta_end = trajectory_end_relative - predicted_state[1:4]
        d_lambda_end_m = R_earth * cos_phi0 * delta_end[0]
        d_phi_end_m = R_earth * delta_end[1]
        d_r_end_m = delta_end[2]
        distance_to_end_m = np.sqrt(
            d_lambda_end_m**2 + d_phi_end_m**2 + d_r_end_m**2
        )

        reward, info = self._compute_reward(
            predicted_state,
            ref_next_state=ref_next_rel,
            trajectory_end_relative=trajectory_end_relative,
            distance_to_end_m=distance_to_end_m,
            action=action
        )

        # ------------------------
        # 3. 步数 + done 判断
        # ------------------------
        self.current_step += 1
        # 这里用 1km 作为“到达终点”的判定
        done = (self.current_step >= self.max_steps) or (distance_to_end_m < 1_000.0)

        steps_remaining_ratio = max(self.max_steps - self.current_step, 0) / self.max_steps
        flat_history = np.concatenate(self.history_states)
        obs = np.concatenate([flat_history, trajectory_end_relative, [steps_remaining_ratio]]).astype(np.float32)

        return obs, float(reward), bool(done), info



        # -------------------------------------------------
    # reward 计算：全状态跟踪 + 动作平滑 + 终点 shaping
    # -------------------------------------------------
    def _compute_reward(self, predicted_state, ref_next_state,
                        trajectory_end_relative, distance_to_end_m, action):
        """
        predicted_state: 当前 step 物理模型算出的下一状态 [v, λ_rel, φ_rel, r_rel, ψ, γ]
        ref_next_state:  参考下一状态（来自真实轨迹 X[idx+1]，同样是相对坐标）
        distance_to_end_m: 当前点到终点的欧式距离（米）
        action: 当前动作，用于平滑惩罚
        """

        R_earth = float(self.r_start)
        cos_phi0 = np.cos(float(self.latitude_start))

        # ---------- 1. 位置误差（米） ----------
        d_lambda = predicted_state[1] - ref_next_state[1]
        d_phi = predicted_state[2] - ref_next_state[2]
        d_r = predicted_state[3] - ref_next_state[3]

        lambda_error_m = R_earth * cos_phi0 * d_lambda
        phi_error_m = R_earth * d_phi
        r_error_m = d_r

        pos_error_m = np.sqrt(
            lambda_error_m**2 + phi_error_m**2 + r_error_m**2
        )

        # ---------- 2. 速度 & 姿态误差 ----------
        v_error = predicted_state[0] - ref_next_state[0]

        psi_error = predicted_state[4] - ref_next_state[4]
        gamma_error = predicted_state[5] - ref_next_state[5]

        # 角度 wrap 到 [-pi, pi]
        psi_error = (psi_error + np.pi) % (2 * np.pi) - np.pi
        gamma_error = (gamma_error + np.pi) % (2 * np.pi) - np.pi

        att_error = np.sqrt(psi_error**2 + gamma_error**2)

        # ---------- 3. 动作平滑惩罚 ----------
        # 希望动作不要太大，避免 r 剧烈震荡
        action_norm = np.linalg.norm(action)

        # ---------- 4. 距终点 shaping ----------
        # 远处给一个平滑的小奖励，近终点加 bonus
        distance_scale = 50_000.0  # 50 km
        distance_reward = np.exp(-distance_to_end_m / distance_scale)

        end_reward = 0.0
        if distance_to_end_m < 5_000.0:      # 5 km 内
            end_reward = 10.0
        elif distance_to_end_m < 20_000.0:   # 20 km 内
            end_reward = 3.0

        # ---------- 5. 加权组合 ----------
        # 根据你的量纲大致调了一组权重，可以再微调
        w_pos = 1e-4   # 10km 位置误差 => ~ -10
        w_vel = 1e-3   # 100 m/s 速度误差 => ~ -10
        w_att = 1.0    # 1 rad 姿态误差 => ~ -1
        w_act = 0.1    # |action|~1 时给点小惩罚

        state_penalty = (
            w_pos * (pos_error_m ** 2) +
            w_vel * (v_error ** 2) +
            w_att * (att_error ** 2) +
            w_act * (action_norm ** 2)
        )

        reward = -state_penalty + distance_reward + end_reward

        info_dict = {
            # 这里 error 全是“米”或“弧度”，便于画图理解
            "lambda_error": float(lambda_error_m),
            "phi_error": float(phi_error_m),
            "r_error": float(r_error_m),
            "next_state_error": float(pos_error_m),
            "distance_reward": float(distance_reward),
            "end_reward": float(end_reward),
            "reward": float(reward),
        }

        return reward, info_dict



    # -------------------------------------------------
    # 辅助: 当索引越界时构造一个 terminal obs
    # -------------------------------------------------
    def _terminal_observation(self):
        # 用当前历史窗口 + 终点 + 0 剩余步数 构造一个合法的 observation
        if len(self.history_states) == 0:
            # 如果还没 reset 过，构造一个全零的 observation
            obs_dim = self.observation_space.shape[0]
            return np.zeros(obs_dim, dtype=np.float32)

        flat_history = np.concatenate(self.history_states)
        trajectory_end_relative = np.array([
            self.trajectory_end[0] - self.longitude_start,
            self.trajectory_end[1] - self.latitude_start,
            self.trajectory_end[2] - self.r_start
        ], dtype=np.float32)
        steps_remaining_ratio = 0.0

        obs = np.concatenate([flat_history, trajectory_end_relative, [steps_remaining_ratio]]).astype(np.float32)
        return obs
