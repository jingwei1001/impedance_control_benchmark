import pinocchio as pin
import numpy as np
import time
import matplotlib.pyplot as plt
from pinocchio.visualize import MeshcatVisualizer
import example_robot_data
from scipy.spatial.transform import Rotation as R

class AdmittanceController:
    def __init__(self, M, D, dt, init_x):
        self.M = M
        self.D = D
        self.dt = dt

        self.x = np.array(init_x, dtype=float)
        self.dx = np.zeros(6)
        self.ddx = np.zeros(6)

        self.M_inv = np.linalg.inv(M)

    def update(self, x_ref, F_ext, F_des=0.0):
        """
        x_ref : reference pose (6D)
        F_ext : external wrench (6D)
        F_des : desired force along x axis
        """

        # -------- only x-axis admittance --------
        F = np.zeros(6)
        F[0] = F_ext[0] - F_des

        self.ddx = self.M_inv @ (F - self.D @ self.dx)
        self.dx += self.ddx * self.dt
        self.x += self.dx * self.dt

        # -------- other axes strictly follow reference --------
        self.x[1:] = x_ref[1:]
        self.dx[1:] = 0.0

        return self.x.copy()

def external_force(x):
    """
    x: current task-space pose
    """

    F = np.zeros(6)

    x0 = 0.65       # equilibrium point
    k_field = 1200  # field stiffness (environment!)

    # force always points toward x0
    F[0] = -k_field * (x[0] - x0)

    return F

robot = example_robot_data.load("ur5")
viz = MeshcatVisualizer()
robot.setVisualizer(viz)
robot.initViewer()
robot.loadViewerModel()

NQ, NV = robot.model.nq, robot.model.nv
frame_id = robot.model.getFrameId("ee_link")

q = np.array([0.0, -1.0, 1.2, -3.34, -np.pi/2, 0])
dq = np.zeros(NV)

dt = 0.002
sim_time = 20
steps = int(sim_time / dt)

# -------- admittance parameters --------
M = np.diag([2, 1e6, 1e6, 1e6, 1e6, 1e6])
D = np.diag([80, 0, 0, 0, 0, 0])

pin.forwardKinematics(robot.model, robot.data, q)
pin.updateFramePlacement(robot.model, robot.data, frame_id)
oMf = robot.data.oMf[frame_id]

x_init = np.hstack([
    oMf.translation,
    pin.log(oMf.rotation)
])

adm = AdmittanceController(M, D, dt, x_init)

F_des = -10.0   # desired contact force along x

def reference_traj(t):
    return x_init.copy()   # no motion command

viz.display(q)
time.sleep(2)

print("Simulation start ...")

t = 0.0
for i in range(steps):

    pin.forwardKinematics(robot.model, robot.data, q)
    pin.updateFramePlacement(robot.model, robot.data, frame_id)

    oMf = robot.data.oMf[frame_id]
    p = oMf.translation
    r = pin.log(oMf.rotation)
    x_current = np.hstack([p, r])

    x_ref = reference_traj(t)
    F_ext = external_force(x_current)

    x_new = adm.update(x_ref, F_ext, F_des)

    J = pin.computeFrameJacobian(
        robot.model, robot.data, q, frame_id,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )

    xdot = (x_new - x_current) / dt

    dq_cmd = np.linalg.lstsq(J, xdot, rcond=None)[0]

    dq_cmd = np.clip(dq_cmd, -2.0, 2.0)
    q = pin.integrate(robot.model, q, dq_cmd * dt)

    # viz.display(q)
    t += dt
print(x_current)
