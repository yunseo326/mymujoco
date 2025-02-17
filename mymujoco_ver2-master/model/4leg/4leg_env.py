import numpy as np

from gym import utils
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box
import os

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

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

        xml_path = os.path.join(os.path.dirname(__file__), "4leg_robot.xml")

        self.single_observation_space = Box(low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32 )
        self.single_action_space = Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32 )

        self.action_size = 8
        self.action_low = -1.571 
        self.action_high = 1.571

        # Mujoco 환경 초기화
        MujocoEnv.__init__(
            self,
            xml_path,
            5,  # frame_skip 값 설정
            observation_space=self.single_observation_space,
            render_mode="human")

    def step(self, action):
        """환경의 한 단계 진행."""
        self.do_simulation(action, self.frame_skip)
        # frame_skip:  기본값 5, 1초에 5번의 액션을 취함
        observation = self._get_obs()
        reward, done = self._get_reward()
        terminated=False
        info = {}
        
        if self.render_mode ==  "human":
            self.render()
        # 관찰, 보상, 종료 여부, 추가 정보 반환
        return observation, reward,terminated, done, info
    
    def _get_reward(self):
        reward = 1        
        done = False
        reward = np.float32(reward)

        return reward, done
    
    def _get_obs(self):
        """현재 환경 상태의 관찰 값 반환."""
        # 포지션과 속도 데이터를 가져옴
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()
        position = np.float32(position)
        velocity = np.float32(velocity)

        return np.concatenate((position, velocity))

    def reset_model(self):
        """환경을 초기 상태로 재설정."""
        qpos = self.init_qpos 
        qvel = self.init_qvel
        # self.init_qpos,qvel도 self.sim처럼 처음 선언할때 기록되는 것임 

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
