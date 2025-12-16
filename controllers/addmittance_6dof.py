import pinocchio as pin
import numpy as np
import time
import matplotlib.pyplot as plt
from pinocchio.visualize import MeshcatVisualizer
import example_robot_data
from scipy.spatial.transform import Rotation as R

# =============================
#   Admittance Controller
# =============================
class AdmittanceController:
    def __init__(self, M, D, K, dt, init_x=np.zeros(6)):
        self.M = M
        self.D = D
        self.K = K

        self.dt = dt

        self.M_inv = np.linalg.inv(M)
        self.x = init_x
        self.v = np.zeros(6)

    def update(self, x_ref, F_ext):
        spring = self.K.dot(self.x - x_ref)
        damper = self.D.dot(self.v)
        a = self.M_inv.dot(F_ext - spring - damper)
        self.v = self.v + self.dt * a
        self.x = self.x + self.dt * self.v

        return self.x.copy()

# =============================
#          Setup UR5
# =============================
robot = example_robot_data.load("ur5")
viz = MeshcatVisualizer()
robot.setVisualizer(viz)
robot.initViewer()
robot.loadViewerModel()

NQ, NV = robot.model.nq, robot.model.nv
frame_id = robot.model.getFrameId("ee_link")

q = np.array([0.0, -1.0, 1.2, -3.34, -np.pi/2, 0])
dq = np.zeros(NV)
ddq = np.zeros(robot.model.nv)

dt = 0.002
sim_time = 20
steps = int(sim_time / dt)

M = np.diag([2,2,2,  0.6,0.6,0.6])
D = np.diag([40,40,40,  4,4,4])
K = np.diag([80,80,80,  8,8,8])

log_x = []
log_x_new = []
log_x_ref = []
log_force = []
log_offset = []

pin.forwardKinematics(robot.model, robot.data, q)
x_desired = robot.framePlacement(q, frame_id).translation.tolist()+R.from_matrix(robot.framePlacement(q, frame_id).rotation.tolist()).as_euler('xyz').tolist()
adm = AdmittanceController(M, D, K, dt, init_x=x_desired)

def reference_traj(t):
    return np.array(x_desired).copy() + np.array([0.1*np.sin(0.5*t), 0.0, 0.0, 0,0, 0.0])

def external_force(t):
    if 5 < t < 10:
        return np.array([0, 0, 0,  3*np.sin(3*t), 0, 0])
    if 10 < t < 15:
        return np.array([0, 0, 0,  0, 3*np.sin(3*t), 0])
    if 15 < t < 20:
        return np.array([0, 0, 0,  0, 0, 3*np.sin(3*t)])
    return np.zeros(6)

viz.display(q)
time.sleep(3)

print("Simulation start ...")

t = 0
for i in range(steps):

    pin.forwardKinematics(robot.model, robot.data, q)
    pin.updateFramePlacement(robot.model, robot.data, frame_id)

    oMf = robot.data.oMf[frame_id]
    p = oMf.translation
    r = pin.log(oMf.rotation)              # rotation vector

    x_current = np.hstack([p, r])
    x_ref = reference_traj(t)
    F_ext = external_force(t)
    x_new = adm.update(x_ref, F_ext)
    J = pin.computeFrameJacobian(
        robot.model, robot.data, q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )

    dx = x_new - x_current
    xdot = dx / dt   # desired task-space velocity

    lambda_reg = 1e-6
    JtJ = J.T @ J + lambda_reg * np.eye(NV)
    dq_cmd = np.linalg.solve(JtJ, J.T @ xdot)   # solve for joint velocities

    # optional: clamp velocity magnitude to avoid too large joint steps
    max_joint_vel = 2.0  # rad/s, 根据机器人设定调整
    norm = np.linalg.norm(dq_cmd)
    if norm > max_joint_vel:
        dq_cmd = dq_cmd * (max_joint_vel / norm)

    q = pin.integrate(robot.model, q, dq_cmd * dt)


    viz.display(q)

    log_x.append(x_current.copy())
    log_x_ref.append(x_ref.copy())
    log_x_new.append(x_new.copy())
    log_force.append(F_ext.copy())
    log_offset.append((x_new - x_ref).copy())
    t += dt

print("Simulation done.")

log_x = np.array(log_x)
log_x_ref = np.array(log_x_ref)
log_x_new = np.array(log_x_new)
log_force = np.array(log_force)
log_offset = np.array(log_offset)

# =============================
#      Plot result
# =============================
t_array = np.arange(steps) * dt

plt.figure()
# plt.plot(t_array, log_x[:,0], label="adm_x")
# plt.plot(t_array, log_x_new[:,0], '.', label="new")
# plt.plot(t_array, log_x_ref[:,0], '--', label="ref")
plt.plot(t_array, log_offset[:,0]*10, label="offset_x")
plt.plot(t_array, log_force[:,0], label="force_x")
plt.title("X position")
plt.legend()
plt.show()
