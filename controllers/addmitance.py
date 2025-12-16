import numpy as np
from scipy.spatial.transform import Rotation as R


class AdmittanceController:
    """
    Full 6D admittance controller.
    State includes:
        - position x (3)
        - orientation quat q (4)
        - linear velocity v (3)
        - angular velocity w (3)
    Input:
        - x_ref = (p_ref, q_ref)
        - F_ext = (force[3], torque[3])
    Output:
        - x = (p_new, q_new)
    """

    def __init__(self, 
                 M_pos=np.diag([2,2,2]), 
                 D_pos=np.diag([30,30,30]),
                 K_pos=np.diag([0,0,0]),
                 M_rot=np.diag([0.1,0.1,0.1]),
                 D_rot=np.diag([3,3,3]),
                 K_rot=np.diag([0,0,0]),
                 dt=0.001,
                 init_pos=np.zeros(3),
                 init_quat=np.array([1,0,0,0])):
        
        self.dt = dt

        # Admittance parameters
        self.M_pos = M_pos
        self.D_pos = D_pos
        self.K_pos = K_pos
        self.M_rot = M_rot
        self.D_rot = D_rot
        self.K_rot = K_rot

        # State variables
        self.p = init_pos.copy()          # position
        self.q = init_quat.copy()         # quaternion
        self.v = np.zeros(3)              # linear velocity
        self.w = np.zeros(3)              # angular velocity


    def update(self, x_ref, F_ext):
        """
        x_ref: (p_ref(3), q_ref(4))
        F_ext: [Fx, Fy, Fz, Tx, Ty, Tz]

        Returns new (p, q)
        """

        p_ref, q_ref = x_ref
        force = F_ext[:3]
        torque = F_ext[3:]

        # ------------------------------------------------------------------
        # 1) Position admittance
        # ------------------------------------------------------------------
        e_pos = self.p - p_ref   # position error

        # acceleration = M⁻¹ (F - D v - K e)
        a = np.linalg.inv(self.M_pos) @ (force - self.D_pos @ self.v - self.K_pos @ e_pos)

        self.v = self.v + a * self.dt
        self.p = self.p + self.v * self.dt

        # ------------------------------------------------------------------
        # 2) Orientation admittance using rotation vectors
        # ------------------------------------------------------------------
        # Convert current and ref quaternion to rotation vectors
        r_cur = R.from_quat(self.q).as_rotvec()
        r_ref = R.from_quat(q_ref).as_rotvec()

        e_rot = r_cur - r_ref   # rotation error in rotvec space

        # angular acceleration
        alpha = np.linalg.inv(self.M_rot) @ (torque - self.D_rot @ self.w - self.K_rot @ e_rot)

        self.w = self.w + alpha * self.dt
        r_new = r_cur + self.w * self.dt

        # rotvec → quaternion
        self.q = R.from_rotvec(r_new).as_quat()

        return self.p, self.q
