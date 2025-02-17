import gym
import random
import time
import numpy as np
import mujoco
from gym.envs.registration import register
register(id='box_example-v2', entry_point='model.4leg.env_test:Boxenv',)
print(1)

# Gym 환경 생성 (MuJoCo를 사용한 HalfCheetah 환경 예시)
env = gym.make('box_example-v2')  # MuJoCo 기반 환경
print("make")
env.reset()
# 100번 반복하면서 렌더링 및 액션 수행
for _ in range(1000):
    env.render()
    time.sleep(0.01)
    # 랜덤한 액션 생성 (범위를 -1 ~ 1로 고정)
    
    # action = np.random.uniform(-1, 1,env.action_space.shape[0])
    action = np.array([0, 0,0,0,0,0,0,0])

    # action = [random.uniform(-1, 1) for _ in range(env.action_space.shape[0])]
    # print(action,2)
    # 환경에서 한 스텝 진행
    space = env.step(action)
    #print(space)
    # # 완료 여부 확인 및 초기화
    # if done:
    #     env.reset()

# 환경 종료
env.close()

