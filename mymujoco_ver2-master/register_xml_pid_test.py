import gym
import random
import time
import numpy as np
import mujoco
from gym.envs.registration import register
register(id='box_example-v2', entry_point='model.moving_ball:Boxenv',)

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.prev_error = 0
        self.integral = 0

    def compute(self, measured_value, desired_angle, dt):
        """Calculates the PID output given the measured value and time step (dt)."""
        self.error = desired_angle - measured_value
        self.integral += self.error * dt
        self.derivative = (self.error - self.prev_error) / dt if dt > 0 else 0
        self.output = self.kp * self.error + self.ki * self.integral + self.kd * self.derivative
        # print(self.kp * self.error,self.ki * self.integral,self.kd * self.derivative)
        self.prev_error = self.error
        return self.output


# Gym 환경 생성 (MuJoCo를 사용한 HalfCheetah 환경 예시)
env = gym.make('box_example-v2')  # MuJoCo 기반 환경
obs=env.reset()
pid_1 = PIDController(kp=2.0, ki=0.1, kd=1)
pid_2 = PIDController(kp=10.0, ki=0.1, kd=10)
# approxiamtly servo move value
dt = 0.01  # Time step in seconds
prev_ouput_1 = 0
prev_ouput_2 = 0

# 100번 반복하면서 렌더링 및 액션 수행
for i in range(100000):
    env.render()
    time.sleep(0.01)
    # 랜덤한 액션 생성 (범위를 -1 ~ 1로 고정)
    if i <= 100:
        action = np.array([1, 0.524])  # 0.524  = 30 degree
    elif i >= 100:
        action = np.array([1, 1.047])   # 1.047  = 60 degree
    # elif i <= 300:
    #     action = np.array([1, 1.571])   # 1.571  = 90 degree

    # action = np.random.uniform(-1, 1,env.action_space.shape[0])
    # if i<=10:
    #     action = np.array([2, 2])
    # else:
    #     action = np.random.uniform(-1, 1,env.action_space.shape[0])
    # Example usage
    measured_value_1 = obs[0][3]
    measured_value_2 = obs[0][4]
    # print(obs[0][5],obs[0][6])

    output_1 = pid_1.compute(measured_value_1, action[0],dt)
    output_2 = pid_2.compute(measured_value_2, action[1],dt)
    # print(output_1,output_2)
    # print(output_1-prev_ouput_1, output_2-prev_ouput_2)
    # action = np.array([output_1-prev_ouput_1, output_2-prev_ouput_2])

    # if abs(obs[0][5]) >= 10 or abs(obs[0][6])>=10:
    #     output_1, output_2=0,0
    action = np.array([output_1, output_2])
    # prev_ouput_1 = output_1
    # prev_ouput_2 = output_2

    # action = [random.uniform(-1, 1) for _ in range(env.action_space.shape[0])]
    # 환경에서 한 스텝 진행
    obs = env.step(action)

    print(obs[0][3:])
    # # 완료 여부 확인 및 초기화
    # if done:
    #     env.reset()

env.close()

