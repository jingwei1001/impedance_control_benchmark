import numpy as np
import time
import pinocchio as pin
import example_robot_data
from scipy.spatial.transform import Rotation as R
from pinocchio.visualize import MeshcatVisualizer



class AdmittanceControl1D:
    """
    单自由度导纳控制器
    M * x_ddot + D * x_dot = F
    K = 0
    """

    def __init__(self, M=1.0, D=20.0, dt=0.001):
        self.M = M
        self.D = D
        self.dt = dt

        self.x = 0.0
        self.x_dot = 0.0

    def reset(self):
        self.x = 0.0
        self.x_dot = 0.0

    def step(self, force_input):
        """
        force_input: 标量力（目标是收敛到 0）
        return: 标量位移
        """
        # x_ddot = (F - D * x_dot) / M
        x_ddot = (force_input - self.D * self.x_dot) / self.M

        # 积分
        self.x_dot += x_ddot * self.dt
        self.x += self.x_dot * self.dt

        return self.x

class ClearBoard:
    """
    擦黑板主控制逻辑
    """

    def __init__(
        self,
        dt=0.001,
        contact_force_threshold=3.0,
        desired_contact_force=10.0,
        normal_filter_alpha=0.05,
        exit_pos_threshold=0.02,
    ):
        self.dt = dt

        # 状态
        self.in_contact = False

        # 力阈值
        self.contact_force_threshold = contact_force_threshold
        self.desired_contact_force = desired_contact_force

        # 法线
        self.normal = None
        self.normal_filter_alpha = normal_filter_alpha

        # 导纳控制器（单维）
        self.adm = AdmittanceControl1D(dt=dt)

        # 接触起始参考
        self.contact_pos_ref = None

        # 退出阈值
        self.exit_pos_threshold = exit_pos_threshold

    # ---------- 工具函数 ----------

    def _normalize(self, v):
        norm = np.linalg.norm(v)
        if norm < 1e-6:
            return v
        return v / norm

    def _lowpass_normal(self, old_n, new_n):
        n = (1.0 - self.normal_filter_alpha) * old_n + self.normal_filter_alpha * new_n
        return self._normalize(n)

    def _decompose(self, vec, normal):
        """
        将 vec 分解为法线方向标量 + 切线向量
        """
        normal_comp = np.dot(vec, normal)
        tangential = vec - normal_comp * normal
        return normal_comp, tangential

    # ---------- 主 step ----------

    def step(self, target_pos, target_quat, measured_force):
        """
        target_pos: (3,)
        target_quat: (4,) 直接透传
        measured_force: (3,)
        """

        force_norm = np.linalg.norm(measured_force)

        # ===============================
        # 1. 自由空间
        # ===============================
        if not self.in_contact:
            if force_norm < self.contact_force_threshold:
                return target_pos, target_quat
            

            # ---------- 进入接触 ----------
            self.in_contact = True

            # 初始化法线方向（力的反方向）
            init_normal = -measured_force
            self.normal = self._normalize(init_normal)

            # 初始化导纳
            self.adm.reset()

            # 记录接触位置
            self.contact_pos_ref = target_pos.copy()

        # ===============================
        # 2. 接触状态
        # ===============================

        # ---------- 法线更新（滤波） ----------
        measured_normal = -measured_force
        measured_normal = self._normalize(measured_normal)
        self.normal = self._lowpass_normal(self.normal, measured_normal)

        # ---------- 位移分解 ----------
        pos_error = target_pos - self.contact_pos_ref
        pos_n, pos_t = self._decompose(pos_error, self.normal)
        if abs(pos_n) > self.exit_pos_threshold:
            return target_pos, target_quat

        # ---------- 力残差 ----------
        force_n = np.dot(measured_force, self.normal)
        force_residual = self.desired_contact_force - force_n
        # 送入导纳的是“目标为 0 的残差”
        adm_disp = self.adm.step(force_residual)

        # ---------- 合成目标位移 ----------
        new_pos = (
            self.contact_pos_ref
            + pos_t
            + adm_disp * self.normal
        )

        # ---------- 退出判据 ----------
        if abs(pos_n) > self.exit_pos_threshold:
            self.in_contact = False
            self.normal = None
            self.contact_pos_ref = None
            return target_pos, target_quat

        return new_pos, target_quat

dt = 0.002
sim_time = 20.0
steps = int(sim_time / dt)

# -------------------------------
# 黑板参数
# -------------------------------
BOARD_X = 0.7           # 黑板位置
FORCE_GAIN = 1000.0     # 1cm -> 10N

# -------------------------------
# 加载机器人
# -------------------------------
robot = example_robot_data.load("ur5")
viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
robot.setVisualizer(viz)
robot.initViewer()
robot.loadViewerModel()

NQ = robot.model.nq
NV = robot.model.nv
frame_id = robot.model.getFrameId("ee_link")

# -------------------------------
# 初始关节
# -------------------------------
q = np.array([0.0, -1.0, 1.2, -3.34, -np.pi/2, 0.0])
dq = np.zeros(NV)

viz.display(q)
time.sleep(2)

# -------------------------------
# 初始化 clear_board
# -------------------------------
clear_board = ClearBoard(
    dt=dt,
    contact_force_threshold=2.0,
    desired_contact_force=10.0,
    normal_filter_alpha=0.05,
    exit_pos_threshold=0.1,
)

# -------------------------------
# 参考轨迹（擦拭：y 方向往返）
# -------------------------------
pin.forwardKinematics(robot.model, robot.data, q)
pin.updateFramePlacement(robot.model, robot.data, frame_id)
oMf = robot.data.oMf[frame_id]

x0 = oMf.translation.copy()
quat0 = R.from_matrix(oMf.rotation).as_quat()

def reference_traj(t):
    """
    目标轨迹：向前靠近黑板 + y 方向擦拭
    """
    pos = x0.copy()
    if pos[0]< 0.74:
        pos[0] += 0.01                     # 向黑板方向
    pos[1] = 0.15 * np.sin(0.5 * t)   # 擦拭
    return pos, quat0.copy()

# -------------------------------
# 外力模型（只与位移有关）
# -------------------------------
def board_contact_force(position):
    """
    position: ee 世界坐标
    """
    penetration = position[0] - BOARD_X
    if penetration <= 0:
        return np.zeros(3)

    # 力方向：-x
    force_mag = FORCE_GAIN * penetration
    return np.array([-force_mag, 0.0, 0.0])

# -------------------------------
# 日志
# -------------------------------
log_force = []
log_x = []
log_x_cmd = []

print("Simulation start ...")

t = 0.0
for i in range(steps):

    # ---- 正向运动学 ----
    pin.forwardKinematics(robot.model, robot.data, q)
    pin.updateFramePlacement(robot.model, robot.data, frame_id)
    oMf = robot.data.oMf[frame_id]

    p = oMf.translation
    r = pin.log(oMf.rotation)

    # ---- 参考轨迹 ----
    p_ref, quat_ref = reference_traj(t)

    # ---- 外力 ----
    F_ext_pos = board_contact_force(p)
    F_ext = np.zeros(6)
    F_ext[:3] = F_ext_pos

    # ---- 擦黑板控制 ----
    p_cmd, quat_cmd = clear_board.step(
        target_pos=p_ref,
        target_quat=quat_ref,
        measured_force=F_ext_pos,
    )

    # ---- 构造任务空间 ----
    x_current = np.hstack([p, r])
    r_cmd = R.from_quat(quat_cmd).as_rotvec()
    x_cmd = np.hstack([p_cmd, r_cmd])

    dx = x_cmd - x_current
    xdot = dx / dt

    # ---- Jacobian ----
    J = pin.computeFrameJacobian(
        robot.model,
        robot.data,
        q,
        frame_id,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    )

    # ---- 最小二乘求 dq ----
    lambda_reg = 1e-6
    dq_cmd = np.linalg.solve(
        J.T @ J + lambda_reg * np.eye(NV),
        J.T @ xdot,
    )

    # ---- 限幅 ----
    max_joint_vel = 2.0
    norm = np.linalg.norm(dq_cmd)
    if norm > max_joint_vel:
        dq_cmd *= max_joint_vel / norm

    # ---- 积分 ----
    q = pin.integrate(robot.model, q, dq_cmd * dt)

    viz.display(q)

    # ---- 日志 ----
    log_force.append(F_ext_pos.copy())
    log_x.append(p.copy())
    log_x_cmd.append(p_cmd.copy())

    t += dt
