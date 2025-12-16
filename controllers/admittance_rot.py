import numpy as np
from scipy.spatial.transform import Rotation as R
import pinocchio as pin
import example_robot_data
import time

class AdmittanceController:
    def __init__(self, M, D, K, dt,
                 init_pos=np.zeros(3),
                 init_quat=np.array([0, 0, 0, 1])):
        self.M = M
        self.D = D
        self.K = K
        self.dt = dt
        self.M_inv = np.linalg.inv(M)

        self.pos = init_pos.copy()
        self.vel = np.zeros(3)

        self.quat = init_quat / np.linalg.norm(init_quat)
        self.omega = np.zeros(3)

        self.quat_ref_prev = init_quat.copy()

    def update(self, pos_ref, quat_ref, F_ext):

        # ---------- translation ----------
        acc = self.M_inv[:3, :3] @ (
            F_ext[:3]
            - self.K[:3, :3] @ (self.pos - pos_ref)
            - self.D[:3, :3] @ self.vel
        )

        self.vel += self.dt * acc
        self.pos += self.dt * self.vel

        # ---------- ensure reference continuity ----------
        if np.dot(self.quat_ref_prev, quat_ref) < 0:
            quat_ref = -quat_ref
        self.quat_ref_prev = quat_ref.copy()

        # ---------- rotation ----------
        q_err = self.quat_multiply(
            quat_ref,
            self.quat_conjugate(self.quat)
        )

        # shortest arc
        if q_err[3] < 0:
            q_err = -q_err

        # log map
        rot_vec = self.quat_log(q_err)
        angle = np.linalg.norm(rot_vec)

        # 🔥 CRITICAL FIX: remove stiffness near π
        # angle_dead = 0 * np.pi
        # if angle > angle_dead:
        #     e_rot = np.zeros(3)
        # else:
        e_rot = -rot_vec

        alpha = self.M_inv[3:, 3:] @ (
            F_ext[3:]
            - self.K[3:, 3:] @ e_rot
            - self.D[3:, 3:] @ self.omega
        )


        self.omega += self.dt * alpha
        self.quat = self.integrate_quat(self.quat, self.omega, self.dt)

        return self.pos.copy(), self.quat.copy()

    # ---------------- SO(3) tools ----------------

    @staticmethod
    def quat_log(q):
        v = q[:3]
        w = q[3]
        nv = np.linalg.norm(v)
        if nv < 1e-6:
            return np.zeros(3)
        angle = 2 * np.arctan2(nv, w)
        return angle * v / nv

    @staticmethod
    def integrate_quat(q, omega, dt):
        theta = np.linalg.norm(omega) * dt
        if theta < 1e-8:
            return q
        axis = omega / np.linalg.norm(omega)
        dq = np.hstack([
            axis * np.sin(theta / 2),
            np.cos(theta / 2)
        ])
        q_next = AdmittanceController.quat_multiply(dq, q)
        return q_next / np.linalg.norm(q_next)

    @staticmethod
    def quat_multiply(q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ])

    @staticmethod
    def quat_conjugate(q):
        return np.array([-q[0], -q[1], -q[2], q[3]])



def main():
    # 加载机器人
    robot = example_robot_data.load("ur5")
    
    # 初始化可视化
    viz = pin.visualize.MeshcatVisualizer()
    robot.setVisualizer(viz)
    viz.initViewer()
    viz.loadViewerModel()
    
    NQ, NV = robot.model.nq, robot.model.nv
    frame_id = robot.model.getFrameId("tool0")  # UR5通常使用tool0作为末端
    
    # 初始关节位置
    q = np.array([0.0, -1.0, 1.2, -3.34, -np.pi/2, 0])
    
    dt = 0.002
    sim_time = 20
    steps = int(sim_time / dt)
    
    # 导纳控制器参数
    M = np.diag([2, 2, 2, 2, 2, 2])
    D = np.diag([40, 40, 40, 4, 4, 4])
    K = np.diag([80, 80, 80, 8, 8, 8])
    
    # 获取初始末端位姿
    pin.framesForwardKinematics(robot.model, robot.data, q)
    pin.updateFramePlacements(robot.model, robot.data)
    
    oMf = robot.data.oMf[frame_id]
    init_pos = oMf.translation.copy()
    init_rot = oMf.rotation.copy()
    init_quat = R.from_matrix(init_rot).as_quat()  # [x, y, z, w]
    
    print(f"初始位置: {init_pos}")
    print(f"初始四元数: {init_quat}")
    
    # 初始化导纳控制器
    adm = AdmittanceController(M, D, K, dt, init_pos=init_pos, init_quat=init_quat)
    
    # 参考轨迹函数 - 现在返回位置和四元数分开
    def reference_traj(t):
        # 位置参考：在x方向正弦运动
        pos_ref = init_pos.copy() + np.array([0.1 * np.sin(0.5 * t), 0.0, 0.0])
        # 旋转参考：保持不变
        quat_ref = init_quat.copy()
        return pos_ref, quat_ref
    
    # 外部力函数
    def external_force(t):
        if 5 < t < 10:
            return np.array([0, 0, 0, 2 * np.sin(3 * t), 0, 0])
        elif 10 < t < 15:
            return np.array([0, 0, 0, 0, 3 * np.sin(3 * t), 0])
        elif 15 < t < 20:
            return np.array([0, 0, 0, 0, 0, 3 * np.sin(3 * t)])
        else:
            return np.zeros(6)
    
    # 显示初始姿态
    viz.display(q)
    time.sleep(5)
    
    print("Simulation start ...")
    
    # 日志记录
    log_pos = []
    log_quat = []
    log_pos_ref = []
    log_quat_ref = []
    log_pos_new = []
    log_quat_new = []
    log_force = []
    log_offset_pos = []
    log_offset_rot = []
    
    t = 0
    for i in range(steps):
        # 获取当前末端位姿
        pin.framesForwardKinematics(robot.model, robot.data, q)
        pin.updateFramePlacements(robot.model, robot.data)
        
        oMf = robot.data.oMf[frame_id]
        current_pos = oMf.translation.copy()
        current_rot = oMf.rotation.copy()
        current_quat = R.from_matrix(current_rot).as_quat()
        
        # 获取参考轨迹
        pos_ref, quat_ref = reference_traj(t)
        
        # 获取外部力
        F_ext = external_force(t)
        
        # 更新导纳控制器
        pos_new, quat_new = adm.update(pos_ref, quat_ref, F_ext)
        
        # 将四元数转换为旋转矩阵
        rot_new = R.from_quat(quat_new).as_matrix()
        
        # 创建目标位姿
        T_target = pin.SE3(rot_new, pos_new)
        
        # 计算雅可比矩阵
        J = pin.computeFrameJacobian(
            robot.model, robot.data, q, frame_id, pin.LOCAL_WORLD_ALIGNED
        )
        
        # 计算位置误差（线性部分）
        pos_error = pos_new - current_pos
        
        # 计算旋转误差（角部分）
        # 使用对数映射计算旋转误差
        R_error = rot_new @ current_rot.T
        rot_error = pin.log3(R_error)
        
        # 合并误差
        error = np.concatenate([pos_error, rot_error])
        desired_velocity = error/dt
        # 计算期望的速度（比例控制）
        # kp = 1.0  # 比例增益
        # desired_velocity = kp * error
        
        # 使用伪逆求解关节速度
        lambda_reg = 1e-6
        J_pinv = np.linalg.pinv(J.T @ J + lambda_reg * np.eye(NV)) @ J.T
        dq_cmd = J_pinv @ desired_velocity
        
        # 限制关节速度
        max_joint_vel = 2.0  # rad/s
        norm = np.linalg.norm(dq_cmd)
        if norm > max_joint_vel:
            dq_cmd = dq_cmd * (max_joint_vel / norm)
        
        # 更新关节位置
        q = pin.integrate(robot.model, q, dq_cmd * dt)
        
        # 更新显示
        if i % 50 == 0:  # 每50步更新一次显示
            viz.display(q)
        
        # 记录日志
        log_pos.append(current_pos.copy())
        log_quat.append(current_quat.copy())
        log_pos_ref.append(pos_ref.copy())
        log_quat_ref.append(quat_ref.copy())
        log_pos_new.append(pos_new.copy())
        log_quat_new.append(quat_new.copy())
        log_force.append(F_ext.copy())
        log_offset_pos.append((pos_new - pos_ref).copy())
        
        # 计算旋转偏移（四元数角度差）
        angle_diff = 2 * np.arccos(np.abs(np.dot(quat_new, quat_ref)))
        log_offset_rot.append(angle_diff)
        
        # 打印进度
        if i % 500 == 0:
            print(f"Time: {t:.2f}s, Pos error: {np.linalg.norm(pos_error):.4f}, "
                  f"Rot error: {np.linalg.norm(rot_error):.4f}")
        
        t += dt
    
    print("Simulation completed.")
    
    # 转换日志为numpy数组
    log_pos = np.array(log_pos)
    log_pos_ref = np.array(log_pos_ref)
    log_pos_new = np.array(log_pos_new)
    log_force = np.array(log_force)
    log_offset_pos = np.array(log_offset_pos)
    log_offset_rot = np.array(log_offset_rot)
    
    # 绘制结果
    try:
        import matplotlib.pyplot as plt
        
        # 创建时间轴
        time_axis = np.arange(steps) * dt
        
        # 创建图形
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        
        # 位置跟踪
        axes[0, 0].plot(time_axis, log_pos[:, 0], label='Actual X')
        axes[0, 0].plot(time_axis, log_pos_ref[:, 0], '--', label='Reference X')
        axes[0, 0].plot(time_axis, log_pos_new[:, 0], '-.', label='Admittance X')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Position X (m)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(time_axis, log_pos[:, 1], label='Actual Y')
        axes[0, 1].plot(time_axis, log_pos_ref[:, 1], '--', label='Reference Y')
        axes[0, 1].plot(time_axis, log_pos_new[:, 1], '-.', label='Admittance Y')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Position Y (m)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(time_axis, log_pos[:, 2], label='Actual Z')
        axes[1, 0].plot(time_axis, log_pos_ref[:, 2], '--', label='Reference Z')
        axes[1, 0].plot(time_axis, log_pos_new[:, 2], '-.', label='Admittance Z')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Position Z (m)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 位置误差
        pos_error_norm = np.linalg.norm(log_pos - log_pos_ref, axis=1)
        axes[1, 1].plot(time_axis, pos_error_norm)
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Position Error Norm (m)')
        axes[1, 1].grid(True)
        
        # 外部力
        axes[2, 0].plot(time_axis, log_force[:, 3], label='Tx')
        axes[2, 0].plot(time_axis, log_force[:, 4], label='Ty')
        axes[2, 0].plot(time_axis, log_force[:, 5], label='Tz')
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('External Torque (Nm)')
        axes[2, 0].legend()
        axes[2, 0].grid(True)
        
        # 偏移量
        axes[2, 1].plot(time_axis, np.linalg.norm(log_offset_pos, axis=1), label='Position Offset')
        axes[2, 1].plot(time_axis, log_offset_rot, label='Rotation Offset (rad)')
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].set_ylabel('Offset')
        axes[2, 1].legend()
        axes[2, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available. Skipping plots.")
    
    return {
        'pos': log_pos,
        'pos_ref': log_pos_ref,
        'pos_new': log_pos_new,
        'force': log_force,
        'offset_pos': log_offset_pos,
        'offset_rot': log_offset_rot
    }

if __name__ == "__main__":
    results = main()