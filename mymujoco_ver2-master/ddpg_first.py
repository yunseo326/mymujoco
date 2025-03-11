# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gym
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the Atari game"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""

    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 25e3
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""

# 환경 만드는 함수
# capture_video 녹화여부 녹화한다면 True 넣어주면 됨, 녹화를 위해서는 render_mode="rgb_array"를 넣어주면 됨
# 녹화하는게 필요없고 그냥 직접적인 화면만 보이면 되다면 else 에 해당하는 부분
def make_env(env_id, capture_video):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling 아래와 같은 버퍼 저장을 하는 이유는 gpu떄문임 device이동시 버퍼를 사용하면 같이 이동하기때문
        # 즉 나는 변수로 self.action_scale로 해도 아무 상관없지만 뭐.. 내비두어도 아무 상관없으니 일단 현상 유지지
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


if __name__ == "__main__":

    # stable_baselines3 을 통해 버퍼 사용을 위해 버전 체크
    import stable_baselines3 as sb3
    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
            poetry run pip install "stable_baselines3==2.0.0a1"
            """
        )
    
    # 변수 사용위해 args에 저장
    args = tyro.cli(Args)
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # envs 만들기
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.capture_video)])

    #이거는 action space가 continuous 한 경우만 가능하다고 알려주기 위한 코드 없어도 되지만 있어서 나쁠건 없음
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(envs).to(device)
    qf1 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    target_actor = Actor(envs).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    envs.single_observation_space.dtype = np.float32

    # 버퍼 정의
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)

    # 여기서부터 학습 관련 코드 시작이라고 보면 됨 위에는 세팅임임
    for global_step in range(args.total_timesteps):

        if global_step < args.learning_starts:  # 탐험을 위해 actor보다는 random에서 값을 가져와서
            actions = np.array([envs.single_action_space.sample()])
        else:   # action을 actor에서 가져와서
            with torch.no_grad():  
                actions = actor(torch.Tensor(obs).to(device))
                # 노이즈 추가 평균이 0 이고 표준편차가 뒤에 나온 내용인 정규분포에서 샘플 얻어내기
                # actor.action_scale 은 중앙값이다 :  action의 범위 최대 - 최소의 절반값 
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                #clip으로 action의 범위를 벗어나지 않게 해줌
                actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # truncation 과 infos 는 필요없어서 반환값이 false 랑 {}임 필요없음 - 그래도 나중의 수정을 위해 일단 남겨둠
        # 나중에 필요없다는게 확실시된다면 ,_,_로 처리해버릴것


        # 화면 보이게 하기
        envs.render()

        """
        강제 종료 시에 나오는게 truncation 과 디버깅에 필요한게 infos인데 둘다 필요없어서 삭제 
        
        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        """
                

        # 이거는 replaybuffer클래스 내용 알고 있으면 되
        """
        아래와 같은 순서로 저장되기 때문에 학습할때 data.observation 이런식으로 꺼냄
        def add(self, obs, next_obs, action, reward, terminated, info):
            self.observations[self.ptr] = obs
            self.next_observations[self.ptr] = next_obs
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.terminations[self.ptr] = terminated
            self.infos[self.ptr] = info    
        """
        # 버퍼에 값 넣어주기
        rb.add(obs, next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # 알고리즘 학습하는 부분
        if global_step > args.learning_starts:
            # 버퍼에서 랜덤하게 가져오기
            data = rb.sample(args.batch_size)

            # obs 즉 (action, state -> next state 등을 통해)q값 계산
            with torch.no_grad():
                next_state_actions = target_actor(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (qf1_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)

            # q_critic 학습 계산
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            q_optimizer.zero_grad()
            qf1_loss.backward()
            q_optimizer.step()

            # td3 처럼 학습 빈도마다 actor 학습 계산 
            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # soft update 임 : target 네트워크의 파라미터를 천천히 복사하는 것 
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

    # 저장하는 부분
    if args.save_model:
        run_name = 'robot'
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save((actor.state_dict(), qf1.state_dict()), model_path)
        print(f"model saved to {model_path}")
        
    envs.close()
