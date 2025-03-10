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

        self.single_observation_space = Box(low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32 )
        self.single_action_space = Box(low=np.array([-0.611,-1.396,-0.611,-1.396,  -2.182,-1.396,-2.182,-1.396]), 
                                     high=np.array([2.182, 1.396, 2.182, 1.396,   0.611, 1.396, 0.611, 1.396]), 
                                     shape=(8,), 
                                     dtype=np.float32 )

        # 4개의 다리를 가진 로봇의 관찰 공간과 행동 공간 정의 

        forward_reward_weight=1.0
        ctrl_cost_weight=0.1

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight

        # Mujoco 환경 초기화
        MujocoEnv.__init__(
            self,
            xml_path,
            5,  # frame_skip 값 설정
            observation_space=self.single_observation_space,
            render_mode="human")

    def step(self, action):
        """환경의 한 단계 진행."""
        x_position_before = self.data.qpos[0]
        y_position_before = self.data.qpos[1]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        y_position_after = self.data.qpos[1]

        xy_velocity = np.sqrt((x_position_after - x_position_before)**2 + (y_position_after - y_position_before)**2)
        if x_position_before > x_position_after or y_position_before > y_position_after:
            xy_velocity = -xy_velocity
        action = np.float32(action)
        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * xy_velocity

        observation = self._get_obs()
        reward, done = self._get_reward(forward_reward,ctrl_cost,observation[0],observation[1],self.data.qpos[2])
        terminated=False
        
        info = {
            "x_position": x_position_after,
            "x_velocity": xy_velocity,
            "reward_run": forward_reward,
            "reward_ctrl": -ctrl_cost,
        }
        
        if self.render_mode ==  "human":
            self.render()
        # 관찰, 보상, 종료 여부, 추가 정보 반환
        return observation, reward,terminated, done, info
    

    def _get_reward(self,forward_reward,ctrl_cost,x_axis,y_axis,height_z):
        
        # 2. 
        if abs(height_z) < 0.15 and abs(x_axis) < 0.6 and abs(y_axis) < 0.6:
            # health reward 
            reward = forward_reward #- ctrl_cost
        else:
            # unhealth reward  maximum 1.5 = 0.6 + 0.6 + 0.3
            reward = -abs(height_z) -abs(x_axis) -abs(y_axis)

        done = False

        #1. failure 
        if x_axis < -1 or x_axis > 1 or y_axis < -1 or y_axis > 1:
            done = True
            reward = -3

        reward = np.float32(reward)
        return reward, done
    
    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost
    
    def _get_obs(self):
        """현재 환경 상태의 관찰 값 반환."""
        # 포지션과 속도 데이터를 가져옴
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()
        x,y,z = self.quaternion_to_euler(position[4],position[5],position[6],position[3])
        position = np.delete(position,6)
        position[3:6] = [x,y,z]
        position = np.delete(position,[0,1,2])
        self.position = np.float32(position)
        self.velocity = np.float32(velocity)
        return np.concatenate((self.position, self.velocity))

    def quaternion_to_euler(self, x, y, z, w):
        sinr_cosp = 2*(w*x + y*z)
        cosr_cosp = 1 - 2*(x*x + y*y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2*(w*y - z*x)
        pitch = np.arcsin(sinp)

        siny_cosp = 2*(w*z + x*y)
        cosy_cosp = 1 - 2*(y*y + z*z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw
    
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


        