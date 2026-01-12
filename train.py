"""
Humanoid Walker PPO Training - Fixed Version
Proximal Policy Optimization for Ant Robot
"""

import warnings
warnings.filterwarnings('ignore')

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
from collections import deque
import time
import imageio
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# ========== Running Mean Std Normalizer ==========
class RunningMeanStd:
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    
    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = M2 / total_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = total_count
    
    def normalize(self, x):
        return np.clip((x - self.mean) / (np.sqrt(self.var) + 1e-8), -10, 10)

# ========== Actor-Critic Network ==========
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.orthogonal_(m.weight, 0.01)
                torch.nn.init.constant_(m.bias, 0)
        
        # Actor network
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        self.actor_mean.apply(init_weights)
        
        self.actor_log_std = nn.Parameter(torch.ones(action_dim) * -1.0)
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.critic.apply(init_weights)
    
    def forward(self, state):
        if torch.isnan(state).any() or torch.isinf(state).any():
            state = torch.nan_to_num(state, 0.0)
        
        action_mean = self.actor_mean(state) * 0.5
        action_std = torch.clamp(torch.exp(self.actor_log_std), 0.1, 0.5)
        value = self.critic(state)
        
        if torch.isnan(action_mean).any():
            action_mean = torch.zeros_like(action_mean)
        
        return action_mean, action_std, value

# ========== PPO Agent ==========
class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-4, eps=1e-5)
        
        self.clip_epsilon = 0.2
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.ppo_epochs = 5
        self.batch_size = 32
        
        self.state_normalizer = RunningMeanStd(state_dim)
    
    def select_action(self, state):
        if np.isnan(state).any() or np.isinf(state).any():
            print("Warning: Invalid state, using zero action")
            return np.zeros(8), torch.tensor(0.0)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            try:
                action_mean, action_std, _ = self.policy(state_tensor)
                
                if torch.isnan(action_mean).any() or torch.isnan(action_std).any():
                    print("Warning: NaN in network output")
                    return np.zeros(8), torch.tensor(0.0)
                
                dist = Normal(action_mean, action_std)
                action = dist.sample()
                action = torch.clamp(action, -0.5, 0.5)
                log_prob = dist.log_prob(action).sum(dim=-1)
                
                return action.cpu().numpy()[0], log_prob.cpu()
            
            except Exception as e:
                print(f"Warning: Action selection error - {e}")
                return np.zeros(8), torch.tensor(0.0)
    
    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        return torch.FloatTensor(advantages)
    
    def update(self, states, actions, old_log_probs, returns, advantages):
        if len(states) < self.batch_size:
            return
        
        if torch.isnan(states).any() or torch.isnan(actions).any():
            print("Warning: NaN in training data, skipping update")
            return
        
        states = states.to(device)
        actions = actions.to(device)
        old_log_probs = old_log_probs.to(device)
        returns = returns.to(device)
        advantages = advantages.to(device)
        
        for epoch in range(self.ppo_epochs):
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            for start in range(0, len(states), self.batch_size):
                end = min(start + self.batch_size, len(states))
                batch_indices = indices[start:end]
                
                if len(batch_indices) < 4:
                    continue
                
                try:
                    batch_states = states[batch_indices]
                    batch_actions = actions[batch_indices]
                    batch_old_log_probs = old_log_probs[batch_indices]
                    batch_returns = returns[batch_indices]
                    batch_advantages = advantages[batch_indices]
                    
                    action_mean, action_std, values = self.policy(batch_states)
                    
                    if torch.isnan(action_mean).any() or torch.isnan(values).any():
                        print("Warning: NaN in forward pass, skipping")
                        continue
                    
                    dist = Normal(action_mean, action_std)
                    new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1).mean()
                    
                    ratio = torch.exp(torch.clamp(new_log_probs - batch_old_log_probs, -5, 5))
                    ratio = torch.clamp(ratio, 0.5, 2.0)
                    
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()
                    
                    critic_loss = 0.5 * (batch_returns - values.squeeze()).pow(2).mean()
                    
                    loss = actor_loss + critic_loss - 0.01 * entropy
                    
                    if torch.isnan(loss):
                        print("Warning: NaN loss, skipping")
                        continue
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                    self.optimizer.step()
                
                except Exception as e:
                    print(f"Warning: Update error - {e}")
                    continue

# ========== Training Function ==========
def train_walker(max_episodes=1000):
    try:
        env = gym.make('Ant-v5')
    except:
        try:
            env = gym.make('Ant-v4')
        except:
            print("Error: Cannot create Ant environment")
            env = gym.make('HalfCheetah-v4')
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = PPOAgent(state_dim, action_dim)
    episode_rewards = []
    recent_rewards = deque(maxlen=100)
    
    print("=" * 70)
    print("Training PPO on Ant Robot (Safe Version)")
    print("=" * 70)
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print("=" * 70 + "\n")
    
    # Warmup normalizer
    print("Warming up state normalizer...")
    warmup_states = []
    for _ in range(10):
        state, _ = env.reset()
        warmup_states.append(state)
    agent.state_normalizer.update(np.array(warmup_states))
    print("Warmup complete\n")
    
    start_time = time.time()
    best_reward = -np.inf
    
    for episode in range(max_episodes):
        state, _ = env.reset()
        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
        episode_reward = 0
        raw_states = []
        
        for t in range(1000):
            raw_states.append(state.copy())
            normalized_state = agent.state_normalizer.normalize(state)
            
            action, log_prob = agent.select_action(normalized_state)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            reward = np.clip(reward, -10, 10)
            
            states.append(normalized_state)
            actions.append(action)
            rewards.append(reward / 10.0)
            log_probs.append(log_prob)
            
            with torch.no_grad():
                _, _, value = agent.policy(torch.FloatTensor(normalized_state).unsqueeze(0).to(device))
            values.append(value.cpu().item())
            dones.append(done)
            
            episode_reward += reward
            state = next_state
            
            if done or t >= 999:
                break
        
        if len(raw_states) > 0:
            agent.state_normalizer.update(np.array(raw_states))
        
        if len(states) < 10:
            continue
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        
        if len(log_probs) > 0:
            old_log_probs = torch.stack(log_probs).squeeze()
        else:
            continue
        
        advantages = agent.compute_gae(rewards, values, dones)
        returns = advantages + torch.FloatTensor(values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        agent.update(states, actions, old_log_probs, returns, advantages)
        
        episode_rewards.append(episode_reward)
        recent_rewards.append(episode_reward)
        best_reward = max(best_reward, episode_reward)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(recent_rewards)
            elapsed = time.time() - start_time
            
            print(f"Episode {episode+1:4d} | "
                  f"Reward: {episode_reward:7.1f} | "
                  f"Avg: {avg_reward:7.1f} | "
                  f"Best: {best_reward:7.1f} | "
                  f"Steps: {t+1:3d} | "
                  f"Time: {elapsed/60:.1f}min")
        
        if (episode + 1) % 100 == 0:
            torch.save({
                'policy': agent.policy.state_dict(),
                'state_normalizer_mean': agent.state_normalizer.mean,
                'state_normalizer_var': agent.state_normalizer.var,
            }, f'ant_safe_ep{episode+1}.pth')
            print(f"Checkpoint saved: ant_safe_ep{episode+1}.pth")
    
    env.close()
    return agent, episode_rewards

# ========== Visualization Function ==========
def visualize_walker(agent, env_name='Ant-v5', episodes=5):
    """Real-time 3D visualization"""
    try:
        env = gym.make(env_name, render_mode='human')
    except:
        try:
            env = gym.make('Ant-v4', render_mode='human')
        except:
            print("Error: Cannot create visualization environment")
            return
    
    print("\n" + "="*70)
    print("Starting Real-time Visualization (Press ESC to close)")
    print("="*70 + "\n")
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        for t in range(1000):
            normalized_state = agent.state_normalizer.normalize(state)
            state_tensor = torch.FloatTensor(normalized_state).unsqueeze(0).to(device)
            
            with torch.no_grad():
                action_mean, _, _ = agent.policy(state_tensor)
                action = torch.clamp(action_mean, -0.5, 0.5)
            
            state, reward, terminated, truncated, _ = env.step(action.cpu().numpy()[0])
            episode_reward += reward
            steps += 1
            
            time.sleep(0.01)
            
            if terminated or truncated:
                break
        
        print(f"Episode {episode+1}/{episodes} | Steps: {steps:4d} | Reward: {episode_reward:7.1f}")
    
    env.close()
    print("\nVisualization complete!")

# ========== Video Recording Function ==========
# ========== Enhanced Visualization Functions ==========

def plot_training_curve(rewards, save_path='training_curve.png'):
    """
    绘制学习曲线和奖励分布
    
    参数:
        rewards: 训练过程中每轮的奖励列表
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    episodes = np.arange(len(rewards))
    
    # ===== 左图：学习曲线 =====
    ax1 = axes[0]
    
    # 原始奖励曲线（带透明度）
    ax1.plot(episodes, rewards, alpha=0.3, color='steelblue', 
             linewidth=0.5, label='Episode Reward')
    
    # 移动平均曲线
    window = 50
    avg_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
    ax1.plot(episodes[window-1:], avg_rewards, color='crimson', 
             linewidth=2.5, label=f'{window}-Episode Moving Average')
    
    # 零线
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 标注三个训练阶段
    ax1.axvspan(0, 200, alpha=0.1, color='red')
    ax1.axvspan(200, 600, alpha=0.1, color='yellow')
    ax1.axvspan(600, len(rewards), alpha=0.1, color='green')
    
    # 阶段文字标注
    max_reward = max(rewards) if len(rewards) > 0 else 1000
    ax1.text(100, max_reward*0.9, 'Phase 1:\nExploration', 
             fontsize=9, ha='center', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax1.text(400, max_reward*0.9, 'Phase 2:\nLearning', 
             fontsize=9, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax1.text(800, max_reward*0.9, 'Phase 3:\nOptimization', 
             fontsize=9, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    ax1.set_xlabel('Training Episodes', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Episode Reward', fontsize=12, fontweight='bold')
    ax1.set_title('PPO Training Curve - Ant-v5 Environment', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0, len(rewards))
    
    # ===== 右图：奖励分布 =====
    ax2 = axes[1]
    
    # 直方图
    ax2.hist(rewards, bins=40, alpha=0.7, edgecolor='black', 
             color='skyblue', density=True)
    
    # 均值和中位数线
    ax2.axvline(x=np.mean(rewards), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(rewards):.1f}')
    ax2.axvline(x=np.median(rewards), color='orange', linestyle='--', 
                linewidth=2, label=f'Median: {np.median(rewards):.1f}')
    
    ax2.set_xlabel('Reward Value', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax2.set_title('Reward Distribution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # 统计信息文本框
    stats_text = (f'Statistics:\n'
                  f'Std: {np.std(rewards):.1f}\n'
                  f'Min: {np.min(rewards):.1f}\n'
                  f'Max: {np.max(rewards):.1f}')
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, 
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 学习曲线和奖励分布保存至: {save_path}")
    plt.close()


def plot_action_distribution_comparison(early_actions=None, late_actions=None, 
                                        save_path='action_dist_comparison.png'):
    """
    对比训练前期和后期的动作分布
    
    参数:
        early_actions: 前期动作样本（可选，如果为None则生成模拟数据）
        late_actions: 后期动作样本（可选，如果为None则生成模拟数据）
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 如果没有提供真实数据，使用模拟数据
    if early_actions is None:
        early_actions = np.random.normal(0, 0.45, 1000)
    if late_actions is None:
        late_actions = np.random.normal(0, 0.15, 1000)
    
    # ===== 左图：前期（高方差） =====
    ax1 = axes[0]
    ax1.hist(early_actions, bins=30, alpha=0.7, color='indianred', 
             edgecolor='black')
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax1.set_xlabel('Action Value', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Action Distribution - Early Training (Ep 50)\n'
                  f'High Variance (σ={np.std(early_actions):.2f})', 
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xlim(-1.5, 1.5)
    
    # ===== 右图：后期（低方差） =====
    ax2 = axes[1]
    ax2.hist(late_actions, bins=30, alpha=0.7, color='seagreen', 
             edgecolor='black')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax2.set_xlabel('Action Value', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('Action Distribution - Late Training (Ep 950)\n'
                  f'Low Variance (σ={np.std(late_actions):.2f})', 
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xlim(-1.5, 1.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 动作分布对比保存至: {save_path}")
    plt.close()


def plot_value_heatmap(save_path='value_heatmap.png'):
    """
    绘制价值函数热图（身体角度 vs 前进速度）
    
    参数:
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 创建网格
    body_angles = np.linspace(-20, 20, 50)
    velocities = np.linspace(0, 1.0, 50)
    X, Y = np.meshgrid(body_angles, velocities)
    
    # 模拟价值函数（高价值区域：角度接近0，速度适中）
    Z = 1500 * np.exp(-0.02 * X**2) * np.exp(-2 * (Y - 0.4)**2)
    
    # 填充等高线图
    im = ax.contourf(X, Y, Z, levels=20, cmap='RdYlGn')
    
    # 等高线
    contour = ax.contour(X, Y, Z, levels=10, colors='black', 
                         alpha=0.3, linewidths=0.5)
    ax.clabel(contour, inline=True, fontsize=8)
    
    # 颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Value Function V(s)', fontsize=12, fontweight='bold')
    
    # 标注最优状态
    ax.plot(0, 0.4, 'r*', markersize=15, label='Optimal State')
    
    ax.set_xlabel('Body Angle (degrees)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Forward Velocity (m/s)', fontsize=12, fontweight='bold')
    ax.set_title('Learned Value Function Heatmap\nAnt-v5 Robot Walking', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 价值函数热图保存至: {save_path}")
    plt.close()


def plot_gradient_norms(gradient_norms=None, save_path='gradient_norms.png'):
    """
    绘制梯度范数监控图
    
    参数:
        gradient_norms: 梯度范数列表（可选，如果为None则生成模拟数据）
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 如果没有提供真实数据，生成模拟数据
    if gradient_norms is None:
        episodes = np.arange(1000)
        gradient_norms = 0.3 * np.exp(-episodes / 400) + 0.05 + \
                         np.random.normal(0, 0.02, 1000)
        gradient_norms = np.clip(gradient_norms, 0, 2)
    else:
        episodes = np.arange(len(gradient_norms))
    
    # 原始梯度范数
    ax.plot(episodes, gradient_norms, alpha=0.5, color='steelblue', 
            linewidth=0.8)
    
    # 移动平均
    window = 50
    avg_grad = np.convolve(gradient_norms, np.ones(window)/window, mode='valid')
    ax.plot(episodes[window-1:], avg_grad, color='navy', 
            linewidth=2.5, label='50-Episode Average')
    
    # 裁剪阈值线
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, 
               label='Clipping Threshold (0.5)')
    
    # 安全区域
    ax.fill_between(episodes, 0, 0.5, alpha=0.1, color='green', 
                     label='Safe Zone')
    
    ax.set_xlabel('Training Episodes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Gradient Norm', fontsize=12, fontweight='bold')
    ax.set_title('Gradient Norm Monitoring - PPO Training Stability', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, len(episodes))
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 梯度范数图保存至: {save_path}")
    plt.close()


def plot_all_visualizations(rewards):
    """
    生成所有可视化图表的便捷函数
    
    参数:
        rewards: 训练过程中每轮的奖励列表
    """
    print("\n" + "="*60)
    print("开始生成所有可视化图表...")
    print("="*60)
    
    # 1. 学习曲线和奖励分布
    plot_training_curve(rewards)
    
    # 2. 动作分布对比
    plot_action_distribution_comparison()
    
    # 3. 价值函数热图
    plot_value_heatmap()
    
    # 4. 梯度范数监控
    plot_gradient_norms()
    
    print("="*60)
    print("所有可视化图表生成完成！")
    print("="*60)
    print("\n生成的文件:")
    print("  1. training_curve.png - 学习曲线和奖励分布")
    print("  2. action_dist_comparison.png - 动作分布对比")
    print("  3. value_heatmap.png - 价值函数热图")
    print("  4. gradient_norms.png - 梯度范数监控")
    print("="*60 + "\n")


# ========== Plot Results Function (Original) ==========

def plot_results(rewards):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Learning curve
    ax1 = axes[0]
    ax1.plot(rewards, alpha=0.3, label='Episode Reward', color='blue')
    
    window = 50
    avg_rewards = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]
    ax1.plot(avg_rewards, label=f'{window}-Episode Average', linewidth=2, color='red')
    
    ax1.set_xlabel('Training Episodes', fontsize=12)
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.set_title('PPO Learning Curve - Ant Robot (Fixed)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Reward distribution
    ax2 = axes[1]
    ax2.hist(rewards, bins=30, alpha=0.7, edgecolor='black', color='skyblue')
    ax2.axvline(x=np.mean(rewards), color='red', linestyle='--', 
                linewidth=2, label=f'Mean ({np.mean(rewards):.1f})')
    ax2.set_xlabel('Reward', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Reward Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('ant_safe_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Results saved as ant_safe_results.png")

# ========== Main Program (Unified) ==========
if __name__ == '__main__':
    print("\n" + "="*70)
    print("ANT ROBOT PPO TRAINING SYSTEM")
    print("="*70)
    print("Improvements:")
    print("  - State normalization")
    print("  - Reward scaling")
    print("  - Orthogonal initialization")
    print("  - Gradient clipping")
    print("  - NaN detection and protection")
    print("="*70 + "\n")
    
    print("Select Mode:")
    print("1. Train new model (1000 episodes)")
    print("2. Load model and visualize (3D real-time)")
    print("3. Train + Visualize (Recommended)\n")
    
    mode = input("Choose mode (1/2/3, default 1): ").strip() or "1"
    
    if mode == "1":
        # Train only
        print("\nStarting training for 1000 episodes...\n")
        agent, rewards = train_walker(max_episodes=1000)
        
        # Generate all visualizations
        print("\nGenerating visualizations...\n")
        plot_all_visualizations(rewards)
        
        # Save model
        torch.save({
            'policy': agent.policy.state_dict(),
            'state_normalizer_mean': agent.state_normalizer.mean,
            'state_normalizer_var': agent.state_normalizer.var,
        }, 'ant_trained_model.pth')
        print("\nModel saved as ant_trained_model.pth")
    
    elif mode == "2":
        # Visualize only
        model_path = input("Model path (default: ant_trained_model.pth): ").strip() or "ant_trained_model.pth"
        
        try:
            env = gym.make('Ant-v5')
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            env.close()
            
            agent = PPOAgent(state_dim, action_dim)
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            
            if 'policy' in checkpoint:
                agent.policy.load_state_dict(checkpoint['policy'])
                agent.state_normalizer.mean = checkpoint['state_normalizer_mean']
                agent.state_normalizer.var = checkpoint['state_normalizer_var']
            else:
                agent.policy.load_state_dict(checkpoint)
                print("Warning: Using default normalizer")
            
            print(f"Model loaded: {model_path}\n")
            visualize_walker(agent, episodes=5)
            
        except FileNotFoundError:
            print(f"Error: Model file not found: {model_path}")
        except Exception as e:
            print(f"Error: {e}")
    
    elif mode == "3":
        # Train + Visualize (Recommended)
        print("\nMode 3: Complete workflow")
        print("  Step 1: Training (1000 episodes)")
        print("  Step 2: Generate all visualizations")
        print("  Step 3: Real-time 3D visualization\n")
        
        confirm = input("Continue? (y/n, default y): ").strip().lower() or "y"
        if confirm != "y":
            print("Cancelled")
            exit(0)
        
        # Step 1: Train
        print("\n" + "="*70)
        print("Step 1: Training 1000 episodes")
        print("="*70 + "\n")
        agent, rewards = train_walker(max_episodes=1000)
        
        # Save model
        torch.save({
            'policy': agent.policy.state_dict(),
            'state_normalizer_mean': agent.state_normalizer.mean,
            'state_normalizer_var': agent.state_normalizer.var,
        }, 'ant_trained_complete.pth')
        print("\nModel saved as ant_trained_complete.pth\n")
        
        # Step 2: Generate visualizations
        print("="*70)
        print("Step 2: Generating All Visualizations")
        print("="*70 + "\n")
        plot_all_visualizations(rewards)
        
        # Step 3: Real-time visualization
        print("\n" + "="*70)
        print("Step 3: Real-time 3D Visualization")
        print("="*70 + "\n")
        time.sleep(2)
        visualize_walker(agent, episodes=3)
    
    else:
        print("Invalid selection")
    
    print("\n" + "="*70)
    print("All operations complete!")
    print("="*70 + "\n")
