import pinocchio as pin
import numpy as np
import time
import matplotlib.pyplot as plt
from pinocchio.visualize import MeshcatVisualizer
import example_robot_data
from scipy.spatial.transform import Rotation as R

class ScalarAdmittance1D:
    """
    1D admittance:
        M * x_ddot + D * x_dot = F_env - F_des
    """

    def __init__(self, M, D, dt, x_init=0.0):
        self.M = float(M)
        self.D = float(D)
        self.dt = dt

        self.x = float(x_init)
        self.dx = 0.0

    def reset(self, x_init=0.0):
        self.x = float(x_init)
        self.dx = 0.0

    def step(self, F_env, F_des):
        """
        输入:
            F_env : 当前法向环境力 (scalar)
            F_des : 期望法向力 (scalar)
        输出:
            x     : 法向位移 (scalar)
        """
        ddx = (F_env - F_des - self.D * self.dx) / self.M
        self.dx += ddx * self.dt
        self.x  += self.dx * self.dt
        return self.x

class ContactNormalEstimator:
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.normal = None

    @staticmethod
    def _normalize(v):
        n = np.linalg.norm(v)
        return v if n < 1e-6 else v / n

    def reset(self):
        self.normal = None

    def update(self, measured_force):
        """
        measured_force: (3,)
        """
        measured_normal = -measured_force
        measured_normal = self._normalize(measured_normal)

        if self.normal is None:
            self.normal = measured_normal
        else:
            self.normal = self._normalize(
                (1.0 - self.alpha) * self.normal
                + self.alpha * measured_normal
            )
        return self.normal

class ClearBoard:
    """
    Blackboard wiping task with normal-direction admittance
    """

    def __init__(
        self,
        dt,
        contact_force_threshold,
        desired_contact_force,
        normal_filter_alpha,
        exit_pos_threshold,
    ):
        self.dt = dt
        self.in_contact = False

        self.contact_force_threshold = contact_force_threshold
        self.desired_contact_force = desired_contact_force
        self.exit_pos_threshold = exit_pos_threshold

        self.normal_estimator = ContactNormalEstimator(
            alpha=normal_filter_alpha
        )

        # === 保留你原本的参数 ===
        self.adm = ScalarAdmittance1D(
            M=2.0,
            D=40.0,
            dt=dt,
            x_init=0.0,
        )

        self.contact_pos_ref = None

    @staticmethod
    def _decompose(vec, normal):
        n = np.dot(vec, normal)
        t = vec - n * normal
        return n, t

    def step(self, target_pos, target_quat, measured_force):
        force_norm = np.linalg.norm(measured_force)

        # ========= Free space =========
        if not self.in_contact:
            if force_norm < self.contact_force_threshold:
                return target_pos, target_quat

            # ---- enter contact ----
            self.in_contact = True
            self.contact_pos_ref = target_pos.copy()
            self.normal_estimator.reset()
            self.adm.reset(x_init=0.0)

        # ========= In contact =========
        normal = self.normal_estimator.update(measured_force)

        pos_error = target_pos - self.contact_pos_ref
        pos_n, pos_t = self._decompose(pos_error, normal)

        if np.abs(pos_n) > self.exit_pos_threshold:
            self.in_contact = False
            self.contact_pos_ref = None
            return target_pos, target_quat

        force_n = np.dot(measured_force, normal)

        adm_disp = self.adm.step(
            F_env=force_n,
            F_des=self.desired_contact_force,
        )

        new_pos = (
            self.contact_pos_ref
            + pos_t
            + adm_disp * normal
        )

        return new_pos, target_quat


# ==========================================================
# Simulation parameters
# ==========================================================
dt = 0.002
sim_time = 30.0
steps = int(sim_time / dt)

# ==========================================================
# Environment (board) parameters
# ==========================================================
BOARD_X = 0.72
K_ENV = 1000.0

def external_force(position):
    """
    Simple 1D environment force field along x-axis
    """
    F = np.zeros(3)

    if position[0] < BOARD_X:
        return F

    F[0] = -K_ENV * (position[0] - BOARD_X)
    return F

# def external_force(position):
#     F = np.zeros(3)

#     y0 = 0.0
#     k_env = 800.0

#     F[1] = -k_env * (position[1] - y0)
#     return F

def reference_traj(t):
    """
    Geometric reference trajectory:
      - slow approach to board
      - sinusoidal wiping along y
    """
    pos = x_init.copy()
    # if pos[0] < 0.76:
    pos[0] += 0.1 * np.sin(t)
    pos[1] += 0.1 * np.sin(2 * t)
    pos[2] += 0.02 * np.cos(0.4 * t)

    return pos, quat_init.copy()


# ==========================================================
# Robot initialization
# ==========================================================
robot = example_robot_data.load("ur5")
viz = MeshcatVisualizer(
    robot.model,
    robot.collision_model,
    robot.visual_model,
)
robot.setVisualizer(viz)
robot.initViewer()
robot.loadViewerModel()

NQ = robot.model.nq
NV = robot.model.nv
frame_id = robot.model.getFrameId("ee_link")

# Initial joint configuration
q = np.array([0.0, -1.0, 1.2, -3.34, -np.pi/2, 0.0])
dq = np.zeros(NV)

viz.display(q)
time.sleep(2.0)

# ==========================================================
# Task controller
# ==========================================================
clear_board = ClearBoard(
    dt=dt,
    contact_force_threshold=2.0,
    desired_contact_force=-10.0,
    normal_filter_alpha=0.05,
    exit_pos_threshold=0.1,
)

# ==========================================================
# Reference trajectory (wiping motion)
# ==========================================================
pin.forwardKinematics(robot.model, robot.data, q)
pin.updateFramePlacement(robot.model, robot.data, frame_id)
oMf = robot.data.oMf[frame_id]

x_init = oMf.translation.copy()
quat_init = R.from_matrix(oMf.rotation).as_quat()

# ==========================================================
# Logging
# ==========================================================
log_force = []
log_pos = []
log_pos_cmd = []


# ==========================================================
# Main simulation loop
# ==========================================================
print("Simulation start ...")
time.sleep(3)
t = 0.0
for k in range(steps):

    # ------------------------------------------------------
    # Forward kinematics
    # ------------------------------------------------------
    pin.forwardKinematics(robot.model, robot.data, q)
    pin.updateFramePlacement(robot.model, robot.data, frame_id)
    oMf = robot.data.oMf[frame_id]

    p = oMf.translation.copy()
    r = pin.log(oMf.rotation)

    # ------------------------------------------------------
    # Reference (geometric)
    # ------------------------------------------------------
    p_ref, quat_ref = reference_traj(t)

    # ------------------------------------------------------
    # Environment force
    # ------------------------------------------------------
    F_ext = external_force(p)

    # ------------------------------------------------------
    # Task-level controller (contact + admittance)
    # ------------------------------------------------------
    # print("ref",p_ref,F_ext)
    p_cmd, quat_cmd = clear_board.step(
        target_pos=p_ref,
        target_quat=quat_ref,
        measured_force=F_ext,
    )
    # print("cmd",p_cmd)

    # ------------------------------------------------------
    # Task-space tracking (velocity IK)
    # ------------------------------------------------------
    x_current = np.hstack([p, r])
    r_cmd = R.from_quat(quat_cmd).as_rotvec()
    x_cmd = np.hstack([p_cmd, r_cmd])

    xdot = (x_cmd - x_current) / dt

    J = pin.computeFrameJacobian(
        robot.model,
        robot.data,
        q,
        frame_id,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    )

    # damped least squares
    lambda_reg = 1e-6
    dq_cmd = np.linalg.solve(
        J.T @ J + lambda_reg * np.eye(NV),
        J.T @ xdot,
    )

    # joint velocity limit
    max_vel = 2.0
    v_norm = np.linalg.norm(dq_cmd)
    if v_norm > max_vel:
        dq_cmd *= max_vel / v_norm

    # integrate
    q = pin.integrate(robot.model, q, dq_cmd * dt)

    # viz.display(q)

    # ------------------------------------------------------
    # Logging
    # ------------------------------------------------------
    log_force.append(F_ext.copy())
    log_pos.append(p.copy())
    log_pos_cmd.append(p_cmd.copy())

    t += dt

log_force = np.array(log_force)

# =============================
#      Plot result
# =============================
t_array = np.arange(steps) * dt

plt.figure()
# plt.plot(t_array, log_x[:,0], label="adm_x")
# plt.plot(t_array, log_x_new[:,0], '.', label="new")
# plt.plot(t_array, log_x_ref[:,0], '--', label="ref")
plt.plot(t_array, log_force[:,0], label="force_x")
# plt.plot(t_array, log_force[:,1], label="force_y")
# plt.plot(t_array, log_force[:,2], label="force_z")
# plt.plot(t_array, np.linalg.norm(log_force, axis=1), label="force_magnitude")
plt.title("Force")
plt.legend()
plt.show()
