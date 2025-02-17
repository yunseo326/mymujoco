import numpy as np

from gym import utils
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box
import os
import random
import math

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}
# after_position=np.array([.2, .2, .2,])
# now_position = np.array([0.0,0.0,1.0])
class Boxenv(MujocoEnv, utils.EzPickle):
    """커스텀 환경 클래스 정의."""
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(self):
        utils.EzPickle.__init__(self)
        # 관찰 공간 정의 (포지션 값 포함 여부에 따라 크기 변경)

        xml_path = os.path.join(os.path.dirname(__file__), "moving_ball.xml")

        self.single_observation_space = Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        self.count=0

        MujocoEnv.__init__(
            self,
            xml_path,
            5,  # frame_skip 값 설정
            observation_space=self.single_observation_space,
            render_mode="human")
            #default_camera_config=DEFAULT_CAMERA_CONFIG,
        #)

    def step(self, action):
        """환경의 한 단계 진행."""
        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()
        reward,done = self._get_reward()
        terminated=False
        info = {}
        
        if self.render_mode ==  "human":
            self.render()
        # 관찰, 보상, 종료 여부, 추가 정보 반환
        return observation, reward,done,terminated, info
    
    def _get_reward(self):
        """현재 환경 상태의 관찰 값 반환."""
        distance_x = abs(self.position[0]-self.position[3])
        distance_y = abs(self.position[1]-self.position[4])
        distance = math.sqrt(distance_x**2 + distance_y**2)        
        reward = distance*(-0.5)
        done = False
        # if abs(self.position[3]) >=2 or abs(self.position[4]) >=2:
        #     print("limitation")
        #     reward = -5
        #     done = True
        if distance_x < .5 and distance_y < .5:
            
            # print("success")
            reward = 5.0
            done = True
        
        reward = np.float32(reward)
        return reward, done
    
    def _get_obs(self):
        """현재 환경 상태의 관찰 값 반환."""
        # 포지션과 속도 데이터를 가져옴
        self.position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()
        # print(self.data.actuator_force) # applied torque (가해지는 힘)
        # print(self.data.qfrc_smooth[6],self.data.qfrc_constraint[6],1)  # total torque...? (qfrc_smooth == 저항 고려하여 실제로 시뮬레이션에 적용되고 있는 힘)(qfrc_constrain == 반작용)
        # print(self.data.cfrc_ext[1],)   # mirror torque (저항 고려하여 반작용)
        self.position = np.float32(self.position)
        velocity = np.float32(velocity)
        self.position = self.position[[0,1,2,7,8]]
        velocity = velocity[[6,7]]
        """
        qpos
        0 : free joint x point
        1 : free joint y point
        2 : free joint z point 

        3 : free joint x angle
        4 : free joint y angle
        5 : free joint z angle 
        6 : free joint w angle (quatenion)

        7 : slide joint x point
        8 : slide joint y point

        qvel 
        0 : free joint x point speed
        1 : free joint y point speed
        2 : free joint z point speed

        3 : free joint x angle speed
        4 : free joint y angle speed
        5 : free joint z angle speed

        6 : slide joint x point speed
        7 : slide joint y point speed
        """
        
        return np.concatenate((self.position, velocity))

    def reset_model(self):
        """환경을 초기 상태로 재설정."""
        # 초기 포지션과 속도에 노이즈 추가
        qpos = self.init_qpos 
        qpos[[0,1]] = np.random.uniform(-2,2,2) 
        while True:
            if abs(qpos[0]-qpos[7]) > 0.5 and abs(qpos[1]-qpos[8]) > 0.5:
                print("again")
                break
            qpos[[0,1]] = np.random.uniform(-2,2,2) 

        qvel = self.init_qvel
        # 상태 설정
        self.set_state(qpos, qvel)
        print("reset")
        # 초기 관찰 값 반환
        return self._get_obs()
    
    def relocate(self):
        """환경을 초기 상태로 재설정."""
        # 초기 포지션과 속도에 노이즈 추가
        qpos = self.data.qpos.flat.copy()
        qpos[[0,1]] = np.random.uniform(-2,2,2) 
        while True:
            if abs(qpos[0]-qpos[7]) >= 0.1 and abs(qpos[1]-qpos[8]) >= 0.1:
                print("again")
                break
            qpos[[0,1]] = np.random.uniform(-2,2,2) 

        qvel = self.data.qvel.flat.copy()

        # 상태 설정
        self.set_state(qpos, qvel)
        
    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)