import numpy as np

from gym import utils
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box
import os
import random
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

        xml_path = os.path.join(os.path.dirname(__file__), "ball.xml")

        self.single_observation_space = Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)
        self.single_action_space = Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        
        self.action_size = 2
        self.action_low = -1.0
        self.action_high = 1.0
        self.count=0
        # Mujoco 환경 초기화
        MujocoEnv.__init__(
            self,
            xml_path,
            5,  # frame_skip 값 설정
            observation_space=self.single_observation_space,render_mode="human")
            #default_camera_config=DEFAULT_CAMERA_CONFIG,
        #)

    def step(self, action):
        """환경의 한 단계 진행."""
        # global after_position,now_position
        # print(after_position)
        # 주어진 액션으로 시뮬레이션 실행
        self.do_simulation(action, self.frame_skip)
        # frame_skip:  기본값 5, 1초에 5번의 액션을 취함
        
        # self.agnet_position = self.data.geom_xpos[3]

        observation = self._get_obs()
        reward,done = self._get_reward()
        terminated=False
        info = {}
        
        if self.render_mode ==  "human":
            #print("render")
            self.render()
        # 관찰, 보상, 종료 여부, 추가 정보 반환
        return observation, reward,done,terminated, info
    
    def _get_reward(self):
        """현재 환경 상태의 관찰 값 반환."""

        reward = 0.1
        done = False
        # print(self.position[2])
        if abs(self.position[2])<1.3:
            reward = -5.0
            self.count+=1
            if self.count == 2:
                self.count=0
                done = True

        reward = np.float32(reward)
        # print(reward.dtype)
        return reward, done
    
    def _get_obs(self):
        """현재 환경 상태의 관찰 값 반환."""
        # 포지션과 속도 데이터를 가져옴
        self.position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()
        # print(self.position)
        self.position = np.float32(self.position)
        velocity = np.float32(velocity)
        # print(self.position.shape, velocity.shape)
        return np.concatenate((self.position, velocity))

    def reset_model(self):
        """환경을 초기 상태로 재설정."""
        # 초기 포지션과 속도에 노이즈 추가
        qpos = self.init_qpos 
        #print(qpos)
        #print(qpos[[0,1]])
        qpos[[0,1]] = np.random.uniform(-.2,.2,2)  # position of ball xy plane
        qpos[[7,8]] = np.random.uniform(-.5,.5,2)  # angle of agent 

        qvel = self.init_qvel
        # self.init_qpos,qvel도 self.sim처럼 처음 선언할때 기록되는 것임 
        #print(qpos,2)
        # 상태 설정
        self.set_state(qpos, qvel)

        # 초기 관찰 값 반환
        return self._get_obs()

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)