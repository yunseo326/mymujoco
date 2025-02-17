import random
import mujoco_py

# MuJoCo 환경 초기화 (XML 파일을 이용한 환경 생성)
model = mujoco_py.load_model_from_path('model/humanoid.xml')  # XML 경로를 여기에 넣으세요
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)

# 100번 반복하면서 렌더링만 수행
for _ in range(10000):
    # 환경 렌더링
    viewer.render()

    # 랜덤한 액션 생성 (예시: 1D 행동 공간을 가진 경우)
    action = [random.uniform(-1, 1) for _ in range(sim.model.nu)]
    
    # 시뮬레이션 한 스텝 진행 (토크 제어 방식으로 액션 적용)
    sim.data.ctrl[:] = action
    sim.step()

# 시뮬레이션 종료
sim.close()

