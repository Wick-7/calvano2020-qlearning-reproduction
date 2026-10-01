import numpy as np
from scipy.optimize import fsolve, minimize_scalar
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from typing import NamedTuple, List, Tuple


# ==========================================
# 1. 参数设置
# ==========================================
class Params(NamedTuple):
    total_episodes: int = 5000000  # 最大模拟步数
    alpha: float = 0.15
    gamma: float = 0.95
    beta: float = 4e-6
    n_agents: int = 2
    m_prices: int = 15
    xi: float = 0.1
    k_memory: int = 1

    a: float = 2.0
    c: float = 1.0
    mu: float = 0.25
    a0: float = 0.0

    # 新增：实验批次设置
    n_sessions: int = 1000  # 训练 100 对
    seed: int = 42


params = Params()


# ==========================================
# 2. 环境与智能体 (保持核心逻辑不变)
# ==========================================
class CalvanoEnv:
    def __init__(self, p: Params, seed=None):
        self.p = p
        self.rng = np.random.default_rng(seed)
        self.p_nash = self._solve_nash()
        self.p_monopoly = self._solve_monopoly()
        diff = self.p_monopoly - self.p_nash
        self.price_grid = np.linspace(
            self.p_nash - self.p.xi * diff,
            self.p_monopoly + self.p.xi * diff,
            self.p.m_prices
        )
        self.state_size = self.p.m_prices ** (self.p.n_agents * self.p.k_memory)
        self.action_size = self.p.m_prices

    def _get_demand(self, p_vec):
        exp_terms = np.exp((self.p.a - p_vec) / self.p.mu)
        exp_0 = np.exp(self.p.a0 / self.p.mu)
        return exp_terms / (np.sum(exp_terms) + exp_0)

    def _solve_nash(self):
        def foc(price):
            p_vec = np.full(self.p.n_agents, price)
            q = self._get_demand(p_vec)[0]
            return price - self.p.c - self.p.mu / (1 - q)

        return fsolve(foc, 1.5)[0]

    def _solve_monopoly(self):
        def objective(price):
            p_vec = np.full(self.p.n_agents, price)
            q = self._get_demand(p_vec)
            return -np.sum((p_vec - self.p.c) * q)

        return minimize_scalar(objective, bounds=(self.p.c, 3.0), method='bounded').x

    def step(self, action_indices):
        p_vec = self.price_grid[np.array(action_indices)]
        demands = self._get_demand(p_vec)
        profits = (p_vec - self.p.c) * demands
        next_state_idx = self._actions_to_state(action_indices)
        return next_state_idx, profits

    def _actions_to_state(self, actions):
        idx = 0
        for i, a in enumerate(actions):
            idx += a * (self.p.m_prices ** i)
        return int(idx)

    def reset(self):
        random_actions = self.rng.integers(0, self.p.m_prices, self.p.n_agents)
        return self._actions_to_state(random_actions)


class CalvanoAgent:
    def __init__(self, env: CalvanoEnv, params: Params, agent_id: int, seed=None):
        self.env = env
        self.p = params
        self.id = agent_id
        self.rng = np.random.default_rng(seed)
        self.q_table = np.zeros((env.state_size, env.action_size))
        self._initialize_q_table()

    def _initialize_q_table(self):
        for a_i in range(self.env.action_size):
            total_profit = 0
            count = 0
            for a_opponent in range(self.env.action_size):
                if self.id == 0:
                    actions = [a_i, a_opponent]
                else:
                    actions = [a_opponent, a_i]
                prices = self.env.price_grid[np.array(actions)]
                demands = self.env._get_demand(prices)
                profit = (prices[self.id] - self.p.c) * demands[self.id]
                total_profit += profit
                count += 1
            expected_profit = total_profit / count
            self.q_table[:, a_i] = expected_profit / (1 - self.p.gamma)

    def choose_action(self, state_idx, t):
        epsilon = np.exp(-self.p.beta * t)
        if self.rng.random() < epsilon:
            return self.rng.integers(0, self.env.action_size)
        else:
            q_values = self.q_table[state_idx, :]
            max_q = np.max(q_values)
            actions_with_max_q = np.flatnonzero(q_values == max_q)
            return self.rng.choice(actions_with_max_q)

    def update(self, state, action, reward, next_state):
        old_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state, :])
        new_q = (1 - self.p.alpha) * old_q + self.p.alpha * (reward + self.p.gamma * max_next_q)
        self.q_table[state, action] = new_q


# ==========================================
# 3. 单次训练函数 (用于并行调用)
# ==========================================
def train_single_session(session_id: int, p: Params):
    # 为每个 session 设置独立的随机种子
    session_seed = p.seed + session_id
    env = CalvanoEnv(p, seed=session_seed)
    agents = [CalvanoAgent(env, p, i, seed=session_seed + i * 1000) for i in range(p.n_agents)]

    state = env.reset()
    history_p0 = []
    history_p1 = []
    record_interval = 1000

    # 收敛检测
    convergence_threshold = 100000
    stable_steps = 0
    last_policies = [np.zeros(env.state_size) for _ in range(p.n_agents)]
    converged = False

    # 训练循环
    for t in range(p.total_episodes):
        actions = [agent.choose_action(state, t) for agent in agents]
        next_state, rewards = env.step(actions)

        for i, agent in enumerate(agents):
            agent.update(state, actions[i], rewards[i], next_state)
        state = next_state

        # 记录数据
        if t % record_interval == 0:
            prices = env.price_grid[np.array(actions)]
            history_p0.append(prices[0])
            history_p1.append(prices[1])

        # 策略稳定性检测 (仅每100步检测一次以加速计算，不影响大局)
        if t % 100 == 0:
            current_policies = [np.argmax(agent.q_table, axis=1) for agent in agents]
            policy_unchanged = True
            for i in range(p.n_agents):
                if not np.array_equal(current_policies[i], last_policies[i]):
                    policy_unchanged = False
                    break

            if policy_unchanged:
                stable_steps += 100  # 因为每100步检测一次
            else:
                stable_steps = 0
                last_policies = current_policies

            if stable_steps >= convergence_threshold:
                converged = True
                break

    return history_p0, history_p1, agents, env


# ==========================================
# 4. 辅助函数：静态最佳应对
# ==========================================
def get_best_response_index(env, rival_price_idx, agent_id=0):
    rival_price = env.price_grid[rival_price_idx]
    best_profit = -np.inf
    best_action = 0
    for action in range(env.action_size):
        if agent_id == 0:
            p_vec = env.price_grid[np.array([action, rival_price_idx])]
        else:
            p_vec = env.price_grid[np.array([rival_price_idx, action])]
        demands = env._get_demand(p_vec)
        profit = (p_vec[agent_id] - env.p.c) * demands[agent_id]
        if profit > best_profit:
            best_profit = profit
            best_action = action
    return best_action


# ==========================================
# 5. 主程序：批量运行与平均分析
# ==========================================
if __name__ == "__main__":
    print(f"开始并行训练 {params.n_sessions} 对智能体 (这可能需要一些时间)...")

    # 1. 并行训练
    # n_jobs=-1 使用所有 CPU 核心
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(train_single_session)(i, params) for i in range(params.n_sessions)
    )

    print("训练完成，开始处理数据...")

    # 解包结果
    # results 列表结构: [(h0, h1, agents, env), (h0, h1, agents, env), ...]
    all_hist_p0 = [r[0] for r in results]
    all_hist_p1 = [r[1] for r in results]
    all_agent_pairs = [r[2] for r in results]
    all_envs = [r[3] for r in results]

    # 2. 处理价格趋势图 (Data Padding & Averaging)
    # 找出最长的历史记录长度
    max_len = max(len(h) for h in all_hist_p0)

    # 创建矩阵用于存储对齐后的数据 (n_sessions, max_len)
    matrix_p0 = np.zeros((params.n_sessions, max_len))
    matrix_p1 = np.zeros((params.n_sessions, max_len))

    for i in range(params.n_sessions):
        h0 = all_hist_p0[i]
        h1 = all_hist_p1[i]
        # 填充有效数据
        matrix_p0[i, :len(h0)] = h0
        matrix_p1[i, :len(h1)] = h1
        # 如果该实验提前收敛，用最后的价格填充剩余部分 (Padding)
        # 这代表"收敛后价格保持不变"
        if len(h0) < max_len:
            matrix_p0[i, len(h0):] = h0[-1]
            matrix_p1[i, len(h1):] = h1[-1]

    # 计算平均趋势
    mean_p0 = np.mean(matrix_p0, axis=0)
    mean_p1 = np.mean(matrix_p1, axis=0)

    # 绘制平均训练趋势图
    plt.figure(figsize=(12, 6))
    x_axis = np.arange(max_len) * 1000  # 还原为真实步数
    plt.plot(x_axis, mean_p0, color='red', label='Agent 0 Avg Price')
    plt.plot(x_axis, mean_p1, color='blue', linestyle='--', label='Agent 1 Avg Price')

    # 参考线
    ref_env = all_envs[0]
    plt.axhline(y=ref_env.p_nash, color='gray', linestyle=':', label='Nash Equilibrium')
    plt.axhline(y=ref_env.p_monopoly, color='green', linestyle=':', label='Monopoly Price')

    plt.title(f'Average Price Evolution over {params.n_sessions} Sessions')
    plt.xlabel('Training Episodes')
    plt.ylabel('Average Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # 3. 批量验证平均惩罚机制 (Averaged Impulse Response)
    print("开始计算平均冲激响应 (Impulse Response)...")
    steps_ir = 20
    ir_p0_matrix = []  # 存储所有 Agent 0 (背叛者) 的轨迹
    ir_p1_matrix = []  # 存储所有 Agent 1 (惩罚者) 的轨迹

    # 遍历所有训练好的智能体对
    for i in tqdm(range(params.n_sessions), desc="Analyzing IR"):
        agents = all_agent_pairs[i]
        env = all_envs[i]

        # A. 找稳态
        state = env.reset()
        # 快速跑100步让其进入稳态
        for _ in range(100):
            actions = [np.argmax(agent.q_table[state]) for agent in agents]
            state, _ = env.step(actions)
        steady_actions = actions

        # B. 模拟背叛过程
        traj_p0 = []
        traj_p1 = []
        current_state = state

        # T=0
        prices = env.price_grid[np.array(steady_actions)]
        traj_p0.append(prices[0])
        traj_p1.append(prices[1])

        for t in range(1, steps_ir + 1):
            intended = [np.argmax(agent.q_table[current_state]) for agent in agents]
            actual = intended.copy()

            # T=1 强制背叛
            if t == 1:
                dev_action = get_best_response_index(env, steady_actions[1], agent_id=0)
                actual[0] = dev_action

            p_vec = env.price_grid[np.array(actual)]
            traj_p0.append(p_vec[0])
            traj_p1.append(p_vec[1])

            next_s, _ = env.step(actual)
            current_state = next_s

        ir_p0_matrix.append(traj_p0)
        ir_p1_matrix.append(traj_p1)

    # 计算平均轨迹
    mean_ir_p0 = np.mean(np.array(ir_p0_matrix), axis=0)
    mean_ir_p1 = np.mean(np.array(ir_p1_matrix), axis=0)

    # 绘制平均惩罚机制图
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(mean_ir_p0)), mean_ir_p0, 'r-o', label='Deviating Agent (Avg)')
    plt.plot(range(len(mean_ir_p1)), mean_ir_p1, 'b--^', label='Non-deviating Agent (Avg)')

    plt.axhline(y=ref_env.p_nash, color='gray', linestyle=':', label='Nash Price')
    plt.axhline(y=ref_env.p_monopoly, color='gray', linestyle='-.', label='Monopoly Price')
    # 绘制稳态价格均值参考线
    plt.axhline(y=mean_ir_p0[0], color='green', alpha=0.3, label='Avg Collusive Price')

    plt.title(f'Averaged Impulse Response (N={params.n_sessions})')
    plt.xlabel('Time (Periods) after Convergence')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
