import pinocchio as pin
import numpy as np
import time
import matplotlib.pyplot as plt
from pinocchio.visualize import MeshcatVisualizer
import example_robot_data
from scipy.spatial.transform import Rotation as R

class AdmittanceController:
    def __init__(self, M, D, dt, init_x = 0.65):
        self.M = M
        self.D = D
        self.dt = dt

        self.x = np.array(init_x, dtype=float)
        self.dx = np.zeros(6)
        self.ddx = np.zeros(6)

        self.M_inv = np.linalg.inv(M)

    def update(self, F_ext, F_des=0.0):
        """
        x_ref : reference pose (6D)
        F_ext : external wrench (6D)
        F_des : desired force along x axis
        """

        # -------- only x-axis admittance --------
        F = np.zeros(6)
        F[0] = F_ext - F_des

        self.ddx = self.M_inv @ (F - self.D @ self.dx)
        self.dx += self.ddx * self.dt
        self.x += self.dx * self.dt

        # -------- other axes strictly follow reference --------
        # self.x[1:] = x_ref[1:]
        # self.dx[1:] = 0.0

        return self.x[0]
    
    def reset(self):
        self.x = 0.65
        self.x_dot = 0.0
    



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

        # 接触起始参考
        self.contact_pos_ref = None

        # 退出阈值
        self.exit_pos_threshold = exit_pos_threshold
        M = np.diag([2, 1e6, 1e6, 1e6, 1e6, 1e6])
        D = np.diag([80, 0, 0, 0, 0, 0])
        self.adm = AdmittanceController(M,D,dt)

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
        measured_normal = -measured_force
        measured_normal = self._normalize(measured_normal)
        self.normal = self._lowpass_normal(self.normal, measured_normal)

        pos_error = target_pos - self.contact_pos_ref
        pos_n, pos_t = self._decompose(pos_error, self.normal)
        if abs(pos_n) > self.exit_pos_threshold:
            return target_pos, target_quat
        
        force_n = np.dot(measured_force, self.normal)
        force_residual = self.desired_contact_force - force_n
        # 送入导纳的是“目标为 0 的残差”
        adm_disp = self.adm.update(force_n,self.desired_contact_force)

        new_pos = (
            self.contact_pos_ref
            + pos_t
            + adm_disp * self.normal
        )

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
    desired_contact_force=-10.0,
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

def external_force(x):
    """
    x: current task-space pose
    """

    F = np.zeros(3)
    if x[0] < 0.65:
        return F

    x0 = 0.65       # equilibrium point
    k_field = 1200  # field stiffness (environment!)

    # force always points toward x0
    F[0] = -k_field * (x[0] - x0)

    return F

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
    F_ext_pos = external_force(p)
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
print(p)