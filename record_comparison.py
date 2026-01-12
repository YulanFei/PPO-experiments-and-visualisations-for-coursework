"""
PPO Training Progress Comparison Video
录制Episode 200, 500, 1000的对比视频

生成三个视频：
1. ep200_demo.mp4 - Episode 200的表现
2. ep500_demo.mp4 - Episode 500的表现  
3. ep1000_demo.mp4 - Episode 1000的表现
4. comparison_grid.mp4 - 三个视频的并排对比（可选）
"""

import warnings
warnings.filterwarnings('ignore')

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import imageio
import os
import glob
from PIL import Image, ImageDraw, ImageFont

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== Running Mean Std Normalizer ==========
class RunningMeanStd:
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    
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
        
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        self.actor_mean.apply(init_weights)
        
        self.actor_log_std = nn.Parameter(torch.ones(action_dim) * -1.0)
        
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
        self.state_normalizer = RunningMeanStd(state_dim)

# ========== Find Models ==========
def find_checkpoint_models():
    """查找不同训练阶段的模型文件"""
    checkpoints = {
        200: None,
        500: None,
        1000: None
    }
    
    # 查找模式
    patterns = {
        200: ['*ep200*.pth', '*200*.pth', '*checkpoint_200*.pth'],
        500: ['*ep500*.pth', '*500*.pth', '*checkpoint_500*.pth'],
        1000: ['*ep1000*.pth', '*1000*.pth', '*checkpoint_1000*.pth', '*final*.pth', '*trained*.pth']
    }
    
    for episode, pattern_list in patterns.items():
        for pattern in pattern_list:
            files = glob.glob(pattern)
            if files:
                checkpoints[episode] = files[0]
                break
    
    return checkpoints

# ========== Record Single Video ==========
def record_single_video(agent, output_file, episode_num, episodes=3, fps=30, max_steps=500):
    """
    录制单个视频
    
    参数:
        agent: PPO智能体
        output_file: 输出文件名
        episode_num: 训练轮数（用于标签）
        episodes: 录制的episode数量
        fps: 帧率
        max_steps: 每个episode最大步数
    """
    print(f"\n录制 Episode {episode_num} 的视频...")
    
    env = gym.make('Ant-v5', render_mode='rgb_array')
    frames = []
    
    total_reward = 0
    total_steps = 0
    
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done and steps < max_steps:
            # 归一化状态
            normalized_state = agent.state_normalizer.normalize(state)
            state_tensor = torch.FloatTensor(normalized_state).unsqueeze(0).to(device)
            
            # 选择动作
            with torch.no_grad():
                try:
                    action_mean, _, _ = agent.policy(state_tensor)
                    action = torch.clamp(action_mean, -0.5, 0.5)
                except:
                    action = torch.zeros(8)
            
            # 执行动作
            state, reward, terminated, truncated, _ = env.step(
                action.cpu().numpy()[0] if torch.is_tensor(action) else action
            )
            episode_reward += reward
            steps += 1
            
            # 获取帧并添加标签
            frame = env.render()
            
            # 转换为PIL图像添加文字
            img = Image.fromarray(frame)
            draw = ImageDraw.Draw(img)
            
            # 添加文字标签
            try:
                # 尝试使用更大的字体
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
            except:
                font = ImageFont.load_default()
            
            label = f"Episode {episode_num} | Step {steps} | Reward: {episode_reward:.0f}"
            
            # 文字背景
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # 绘制半透明背景
            draw.rectangle([(10, 10), (text_width + 30, text_height + 30)], 
                          fill=(0, 0, 0, 180))
            
            # 绘制文字
            draw.text((20, 20), label, fill=(255, 255, 255), font=font)
            
            frames.append(np.array(img))
            
            done = terminated or truncated
        
        total_reward += episode_reward
        total_steps += steps
        print(f"  Episode {ep+1}/{episodes}: {steps} steps, Reward: {episode_reward:.1f}")
    
    env.close()
    
    # 保存视频
    print(f"  保存视频: {output_file}")
    imageio.mimsave(output_file, frames, fps=fps)
    
    avg_reward = total_reward / episodes
    avg_steps = total_steps / episodes
    
    file_size = os.path.getsize(output_file) / (1024*1024)
    print(f"  ✓ 完成: {file_size:.1f}MB, 平均奖励: {avg_reward:.1f}, 平均步数: {avg_steps:.0f}")
    
    return frames, avg_reward, avg_steps

# ========== Create Comparison Grid ==========
def create_comparison_grid(frames_dict, output_file='comparison_grid.mp4', fps=30):
    """
    创建三个视频的并排对比
    
    参数:
        frames_dict: {200: frames, 500: frames, 1000: frames}
        output_file: 输出文件名
        fps: 帧率
    """
    print(f"\n创建对比视频...")
    
    # 获取最短视频长度
    min_length = min(len(frames) for frames in frames_dict.values())
    
    # 裁剪所有视频到相同长度
    for key in frames_dict:
        frames_dict[key] = frames_dict[key][:min_length]
    
    combined_frames = []
    
    for i in range(min_length):
        # 获取三个帧
        frame200 = frames_dict[200][i]
        frame500 = frames_dict[500][i]
        frame1000 = frames_dict[1000][i]
        
        # 调整大小（缩小以便并排放置）
        h, w = frame200.shape[:2]
        new_w = w // 2
        new_h = h // 2
        
        img200 = Image.fromarray(frame200).resize((new_w, new_h), Image.LANCZOS)
        img500 = Image.fromarray(frame500).resize((new_w, new_h), Image.LANCZOS)
        img1000 = Image.fromarray(frame1000).resize((new_w, new_h), Image.LANCZOS)
        
        # 创建网格（2行2列，右下角留空）
        grid = Image.new('RGB', (new_w * 2, new_h * 2), color=(0, 0, 0))
        
        # 放置三个视频
        grid.paste(img200, (0, 0))
        grid.paste(img500, (new_w, 0))
        grid.paste(img1000, (0, new_h))
        
        # 在右下角添加说明
        draw = ImageDraw.Draw(grid)
        
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # 右下角标题
        title = "PPO Training Progress"
        subtitle = "Ant Robot Walking"
        
        draw.text((new_w + 50, new_h + 50), title, fill=(255, 255, 255), font=font_large)
        draw.text((new_w + 50, new_h + 110), subtitle, fill=(200, 200, 200), font=font_small)
        
        # 添加说明
        draw.text((new_w + 50, new_h + 200), "Top Left: Episode 200", fill=(255, 100, 100), font=font_small)
        draw.text((new_w + 50, new_h + 250), "Top Right: Episode 500", fill=(100, 255, 100), font=font_small)
        draw.text((new_w + 50, new_h + 300), "Bottom Left: Episode 1000", fill=(100, 100, 255), font=font_small)
        
        combined_frames.append(np.array(grid))
        
        if (i + 1) % 100 == 0:
            print(f"  处理进度: {i+1}/{min_length} 帧")
    
    # 保存对比视频
    print(f"  保存对比视频: {output_file}")
    imageio.mimsave(output_file, combined_frames, fps=fps)
    
    file_size = os.path.getsize(output_file) / (1024*1024)
    print(f"  ✓ 完成: {file_size:.1f}MB")

# ========== Main Program ==========
def main():
    print("\n" + "="*70)
    print("PPO TRAINING PROGRESS - VIDEO COMPARISON")
    print("="*70)
    print(f"Device: {device}")
    print("="*70)
    
    # 查找模型文件
    print("\n查找训练阶段模型...")
    checkpoints = find_checkpoint_models()
    
    print("\n找到的模型:")
    for ep, model in checkpoints.items():
        status = f"✓ {model}" if model else "✗ 未找到"
        print(f"  Episode {ep:4d}: {status}")
    
    # 检查是否找到所有模型
    missing = [ep for ep, model in checkpoints.items() if model is None]
    if missing:
        print(f"\n警告: 缺少以下阶段的模型: {missing}")
        print("\n请确保你有以下模型文件:")
        print("  - Episode 200: *ep200*.pth 或 *200*.pth")
        print("  - Episode 500: *ep500*.pth 或 *500*.pth")
        print("  - Episode 1000: *ep1000*.pth 或 *1000*.pth 或 *final*.pth")
        
        proceed = input("\n是否继续录制已有的模型? (y/n): ").strip().lower()
        if proceed != 'y':
            return
    
    # 创建环境获取维度
    env = gym.make('Ant-v5')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    env.close()
    
    print(f"\n环境信息:")
    print(f"  状态维度: {state_dim}")
    print(f"  动作维度: {action_dim}")
    
    # 录制参数
    episodes_per_checkpoint = 3
    fps = 30
    max_steps = 500
    
    print(f"\n录制参数:")
    print(f"  每个阶段录制: {episodes_per_checkpoint} episodes")
    print(f"  帧率: {fps} FPS")
    print(f"  最大步数: {max_steps} steps/episode")
    
    # 存储所有帧
    all_frames = {}
    stats = {}
    
    # 录制每个阶段
    for episode_num in [200, 500, 1000]:
        model_path = checkpoints[episode_num]
        
        if model_path is None:
            print(f"\n跳过 Episode {episode_num} (模型不存在)")
            continue
        
        try:
            # 创建智能体
            agent = PPOAgent(state_dim, action_dim)
            
            # 加载模型
            print(f"\n加载模型: {model_path}")
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            
            if 'policy' in checkpoint:
                agent.policy.load_state_dict(checkpoint['policy'])
                agent.state_normalizer.mean = checkpoint['state_normalizer_mean']
                agent.state_normalizer.var = checkpoint['state_normalizer_var']
            else:
                agent.policy.load_state_dict(checkpoint)
                print("  警告: 使用默认normalizer")
            
            # 录制视频
            output_file = f'ep{episode_num}_demo.mp4'
            frames, avg_reward, avg_steps = record_single_video(
                agent=agent,
                output_file=output_file,
                episode_num=episode_num,
                episodes=episodes_per_checkpoint,
                fps=fps,
                max_steps=max_steps
            )
            
            all_frames[episode_num] = frames
            stats[episode_num] = {'reward': avg_reward, 'steps': avg_steps}
            
        except Exception as e:
            print(f"\n错误 (Episode {episode_num}): {e}")
            import traceback
            traceback.print_exc()
    
    # 创建对比视频（如果有至少两个视频）
    if len(all_frames) >= 2:
        # 补充缺失的视频（用黑屏）
        for ep in [200, 500, 1000]:
            if ep not in all_frames:
                # 创建黑屏帧
                dummy_frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(100)]
                all_frames[ep] = dummy_frames
        
        try:
            create_comparison_grid(all_frames, 'comparison_grid.mp4', fps=fps)
        except Exception as e:
            print(f"\n创建对比视频时出错: {e}")
    
    # 打印总结
    print("\n" + "="*70)
    print("视频录制完成！")
    print("="*70)
    print("\n生成的文件:")
    
    for episode_num in [200, 500, 1000]:
        filename = f'ep{episode_num}_demo.mp4'
        if os.path.exists(filename):
            size = os.path.getsize(filename) / (1024*1024)
            reward = stats.get(episode_num, {}).get('reward', 0)
            steps = stats.get(episode_num, {}).get('steps', 0)
            print(f"  {filename:<25s} {size:6.1f}MB  (Avg Reward: {reward:6.1f}, Steps: {steps:5.0f})")
    
    if os.path.exists('comparison_grid.mp4'):
        size = os.path.getsize('comparison_grid.mp4') / (1024*1024)
        print(f"  {'comparison_grid.mp4':<25s} {size:6.1f}MB  (对比视频)")
    
    print("\n" + "="*70)
    print("使用方法:")
    print("  - 播放单个视频: open ep200_demo.mp4")
    print("  - 播放对比视频: open comparison_grid.mp4")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
