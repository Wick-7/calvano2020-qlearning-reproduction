from typing import NamedTuple
import numpy as np
from scipy.optimize import fsolve, minimize_scalar
from tqdm import tqdm
import matplotlib.pyplot as plt

# 参数设置
class Params(NamedTuple):
    total_episodes: int = 5000000  # 最大模拟步数
    alpha: float = 0.15  # 学习率
    gamma: float = 0.95  # 折现因子 delta
    beta: float = 4e-6  # 探索衰减率 Section IV
    n_agents: int = 2  # 双寡头，论文后面探究更多企业可增加
    m_prices: int = 15  # 动作空间大小 m，q_learning需要有限空间
    xi: float = 0.1  # 价格区间扩展系数，网格设置间距
    k_memory: int = 1  # 记忆长度，有限记忆体，k=1足以实现 萝卜大棒策略

    # 需求函数参数,两个企业对称，探究不对称时需要设置不同参数
    a: float = 2.0  # 质量参数 a
    c: float = 1.0  # 边际成本 c
    mu: float = 0.25  # 水平差异化参数 mu
    a0: float = 0.0  # 外部选项质量 a_0
    seed: int = 42

params = Params()
rng = np.random.default_rng(params.seed)

# 设置实验经济环境，伯特兰定价环境
class CalvanoEnv:
    def __init__(self, p: Params):
        self.p = p

        # 计算纳什均衡Nash时的价格 和 合谋垄断Monopoly时的价格
        self.p_nash = self._solve_nash()
        self.p_monopoly = self._solve_monopoly()

        # 构建价格网格，[pN - xi(pM - pN), pM + xi(pM - pN)]，略低于纳什略高于垄断
        diff = self.p_monopoly - self.p_nash
        self.price_grid = np.linspace(
            self.p_nash - self.p.xi * diff,
            self.p_monopoly + self.p.xi * diff,
            self.p.m_prices
        )

        # 状态空间大小为 m^(n*k)，n=2, k=1, 状态空间大小为 15^(2*1) = 225
        self.state_size = self.p.m_prices ** (self.p.n_agents * self.p.k_memory)
        self.action_size = self.p.m_prices

    def _get_demand(self, p_vec):
        """Logit需求函数"""
        exp_terms = np.exp((self.p.a - p_vec) / self.p.mu)
        exp_0 = np.exp(self.p.a0 / self.p.mu)
        denominator = np.sum(exp_terms) + exp_0
        return exp_terms / denominator

    def _solve_nash(self):
        """求解静态Nash均衡"""
        def foc(price):
            # 两个企业对成得p1=p2=price
            p_vec = np.full(self.p.n_agents, price)
            q = self._get_demand(p_vec)[0]  # 两个需求一样取一个就行
            # FOC: p - c - mu / (1 - q) = 0，最优价格满足方程
            return price - self.p.c - self.p.mu / (1 - q)
        return fsolve(foc, 1.5)[0] # 用fsolve在1.5处开始找

    def _solve_monopoly(self):
        """求解垄断合谋时的价格"""
        def objective(price):
            # 由于对称合谋时价格还是一样
            p_vec = np.full(self.p.n_agents, price)
            q = self._get_demand(p_vec)
            profit = np.sum((p_vec - self.p.c) * q)
            return -profit  # 返回负利润
        return minimize_scalar(objective, bounds=(self.p.c, 3.0), method='bounded').x # 求负利润最小值，即最大化利润

    def step(self, action_indices):
        p_vec = self.price_grid[np.array(action_indices)] # 将动作索引转换为价格
        demands = self._get_demand(p_vec) # 获取需求
        profits = (p_vec - self.p.c) * demands # 计算reward
        # 状态更新，对于k=1只记忆上一期来说下一时刻的状态就是当前所有玩家的动作组合
        next_state_idx = self._actions_to_state(action_indices)

        return next_state_idx, profits

    def _actions_to_state(self, actions):
        """将动作组合映射为状态索引"""
        # s = a1 * m^0 + a2 * m^1 ...
        idx = 0
        for i, a in enumerate(actions):
            idx += a * (self.p.m_prices ** i)
        return int(idx)

    def reset(self):
        """随机初始化状态"""
        random_actions = rng.integers(0, self.p.m_prices, self.p.n_agents) # 价格索引0-14随机选择
        return self._actions_to_state(random_actions)


# 设置Q-Learning Agent
class CalvanoAgent:
    def __init__(self, env: CalvanoEnv, params: Params, agent_id: int):
        self.env = env
        self.p = params
        self.id = agent_id
        # 初始化 Q 表
        self.q_table = np.zeros((env.state_size, env.action_size)) # 行代表状态 (State)，列代表动作 (Action)
        self._initialize_q_table() # 论文关键点，特殊设置初始价格

    def _initialize_q_table(self):
        """
        论文公式：Q_i,0(s, a_i) = Sum_{a_-i} [ pi_i(a_i, a_-i) ] / ((1 - delta) * |A|^(n-1))
        假设刚开始，认为对手会随机出价
        """
        # 遍历自己所有可能的动作 a_i
        for a_i in range(self.env.action_size):
            total_profit = 0
            count = 0
            # 遍历对手所有可能的动作 a_-i，这里对手只有一个
            for a_opponent in range(self.env.action_size):
                # 构建价格向量
                if self.id == 0:
                    actions = [a_i, a_opponent]
                else:
                    actions = [a_opponent, a_i]
                prices = self.env.price_grid[np.array(actions)]
                demands = self.env._get_demand(prices)
                profit = (prices[self.id] - self.p.c) * demands[self.id]
                total_profit += profit
                count += 1

            # 计算期望利润并折现
            expected_profit = total_profit / count
            initial_value = expected_profit / (1 - self.p.gamma)

            # 赋值给 Q 表对应的列 (所有状态下，动作 a_i 的初始值都一样)
            self.q_table[:, a_i] = initial_value

    def choose_action(self, state_idx, t):
        """贪心策略结合指数衰减"""
        epsilon = np.exp(-self.p.beta * t) # 随时间增加探索减少
        if rng.random() < epsilon:
            return rng.integers(0, self.env.action_size) # 随机选择价格
        else:
            # 贪婪策略，选Q值最大的动作
            q_values = self.q_table[state_idx, :] # 获取所有Q值
            max_q = np.max(q_values) # 找到最大值
            actions_with_max_q = np.flatnonzero(q_values == max_q) #返回所有最大化的策略
            return rng.choice(actions_with_max_q) # 随机选择，防止每次都选择最低价

    def update(self, state, action, reward, next_state):
        """标准Q-learning更新规则，贝尔曼方程实现"""
        old_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state, :])
        new_q = (1 - self.p.alpha) * old_q + self.p.alpha * (reward + self.p.gamma * max_next_q)
        self.q_table[state, action] = new_q # 更新Q表


# 训练两个agent
def run_simulation():
    env = CalvanoEnv(params)
    print(f"Nash Price: {env.p_nash:.4f}, Monopoly Price: {env.p_monopoly:.4f}")

    agents = [CalvanoAgent(env, params, i) for i in range(params.n_agents)]
    state = env.reset()

    history_p0 = []
    history_p1 = []
    record_interval = 1000 # 每1000次记录数据

    # 检测是否收敛
    convergence_threshold = 100000  # 连续 10万步策略不变
    stable_steps = 0
    # 初始化上一轮的策略 (2个智能体，每个有 state_size 个状态的最优动作)
    # 初始全为0或任意值均可，第一步后会更新
    last_policies = [np.zeros(env.state_size) for _ in range(params.n_agents)]
    converged = False
    max_steps = 10000000  # 设置一个足够大的上限，比如1000万

    pbar = tqdm(range(max_steps), desc="Training")
    for t in pbar:
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

        # 检测策略是否改变
        current_policies = [np.argmax(agent.q_table, axis=1) for agent in agents]
        # 检查两个智能体的策略是否都与上一步完全一致
        # np.array_equal 用于比较两个数组是否完全相同
        policy_unchanged = True
        for i in range(params.n_agents):
            if not np.array_equal(current_policies[i], last_policies[i]):
                policy_unchanged = False
                break
        if policy_unchanged:
            stable_steps += 1
        else:
            stable_steps = 0  # 一旦有变动，计数器重置
            last_policies = current_policies  # 更新 上一轮策略 为当前策略

        # 检查是否达到阈值
        if stable_steps >= convergence_threshold:
            print(f"\n[Converged] Strategies remained unchanged for {stable_steps} steps at episode {t}.")
            converged = True
            break  # 停止训练

        # 更新进度条显示的额外信息
        if t % 10000 == 0:
            pbar.set_postfix({"Stable": stable_steps})
    if not converged:
        print(f"\n[Warning] Reached max steps ({max_steps}) without strict convergence.")

    return agents, env, history_p0, history_p1, record_interval

def get_best_response_index(env, rival_price_idx, agent_id=0):
    """
    获得在静态针对对手价格的最优策略
    """
    rival_price = env.price_grid[rival_price_idx]
    best_profit = -np.inf
    best_action = 0

    # 遍历自己的所有可能动作
    for action in range(env.action_size):
        if agent_id == 0:
            p_vec = env.price_grid[np.array([action, rival_price_idx])]
        else:
            p_vec = env.price_grid[np.array([rival_price_idx, action])]
        # 计算单期利润
        demands = env._get_demand(p_vec)
        profit = (p_vec[agent_id] - env.p.c) * demands[agent_id]
        if profit > best_profit:
            best_profit = profit
            best_action = action
    return best_action


def run_impulse_response_analysis(env, agents, steps=20):
    """
    扰动测试，强制改变agent的策略观察是否采取报复策略
    """
    # 先确保稳定合谋状态
    state = env.reset()
    for _ in range(100):
        actions = [np.argmax(agent.q_table[state]) for agent in agents]
        state, _ = env.step(actions)
    # 记录稳态时的价格索引
    steady_actions = actions
    print(f"稳态价格索引: {steady_actions} -> 价格: {env.price_grid[steady_actions]}")
    # 记录数据
    history_p0 = []
    history_p1 = []
    # 开始模拟Time 0 to Steps
    current_state = state
    # 此时 Time = 0 (Pre-deviation)
    history_p0.append(env.price_grid[steady_actions[0]])
    history_p1.append(env.price_grid[steady_actions[1]])
    print("\n--- 开始背叛测试 ---")
    for t in range(1, steps + 1):
        # 获取双方基于当前状态原本想出的动作 (Greedy)
        intended_actions = [np.argmax(agent.q_table[current_state]) for agent in agents]

        actual_actions = intended_actions.copy()

        # 在 t=1 时强制 Agent0 背叛
        if t == 1:
            # 计算针对对手稳态价格的最佳应对 (背叛)
            deviation_action = get_best_response_index(env, steady_actions[1], agent_id=0)
            actual_actions[0] = deviation_action  # 强制修改动作
            print(
                f"Time {t}: Agent 0 强制背叛! 价格从 {env.price_grid[intended_actions[0]]:.2f} 降至 {env.price_grid[deviation_action]:.2f}")

        # 其他时间让agent继续自主行动
        else:
            # 这里的 actual_actions 就是 Q-table 选出来的
            pass

        # 记录价格
        prices = env.price_grid[np.array(actual_actions)]
        history_p0.append(prices[0])
        history_p1.append(prices[1])

        # 下一步
        next_state, _ = env.step(actual_actions)
        current_state = next_state

        if t <= 5:  # 打印前几轮的详细数据
            print(f"Time {t}: P0={prices[0]:.4f}, P1={prices[1]:.4f}")

    # 绘制扰动背叛测试图
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(history_p0)), history_p0, 'r-o', label='Agent 0')
    plt.plot(range(len(history_p1)), history_p1, 'b--^', label='Agent 1')

    # 画参考线
    plt.axhline(y=env.p_nash, color='gray', linestyle=':', label='Nash Price')
    plt.axhline(y=env.p_monopoly, color='gray', linestyle='-.', label='Monopoly Price')
    plt.axhline(y=history_p0[0], color='green', linestyle='-', alpha=0.3, label='Collusive Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# 运行
if __name__ == "__main__":
    agents, env, hist_p0, hist_p1, interval = run_simulation()
    # 绘制训练过程中的价格收敛趋势图
    plt.figure(figsize=(12, 6))
    # 生成 X 轴坐标 (实际步数 = 记录点索引 * 记录间隔)
    x_axis = np.arange(len(hist_p0)) * interval
    # 绘制两个智能体的价格轨迹
    plt.plot(x_axis, hist_p0, color='red', alpha=0.5, linewidth=1, label='Agent 0 Price')
    plt.plot(x_axis, hist_p1, color='blue', alpha=0.5, linewidth=1, label='Agent 1 Price')
    # 绘制基准线 (Nash 和 Monopoly)
    plt.axhline(y=env.p_nash, color='black', linestyle=':', linewidth=2, label='Nash Equilibrium')
    plt.axhline(y=env.p_monopoly, color='green', linestyle='--', linewidth=2, label='Monopoly Price')
    plt.title(f'Price Convergence (Converged at {x_axis[-1]} steps)')
    plt.xlabel('Training Episodes')
    plt.ylabel('Price')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.show()

    # 结果分析：计算由于合谋带来的利润提升(Delta)
    # 获取最后稳定状态的动作
    final_state = env.reset()
    for _ in range(100):
        actions = [np.argmax(agent.q_table[final_state]) for agent in agents]
        final_state, rewards = env.step(actions)

    final_prices = env.price_grid[np.array(actions)]
    final_profit = rewards[0]  # 对称
    # 计算基准利润
    nash_vec = np.full(params.n_agents, env.p_nash)
    nash_profit = (env.p_nash - params.c) * env._get_demand(nash_vec)[0]
    mon_vec = np.full(params.n_agents, env.p_monopoly)
    mon_profit = (env.p_monopoly - params.c) * env._get_demand(mon_vec)[0]
    # Delta = (Pi_actual - Pi_Nash) / (Pi_Monopoly - Pi_Nash)
    delta = (final_profit - nash_profit) / (mon_profit - nash_profit)

    print(f"\nSimulation Results:")
    print(f"Final Prices: {final_prices}")
    print(f"Calculated Delta (Profit Gain): {delta:.4f} (Goal: > 0.7 for collusion)")

    # 运行分析 (需要传入之前训练好的 env 和 agents)
    run_impulse_response_analysis(env, agents)
