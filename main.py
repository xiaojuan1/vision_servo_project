import cv2
import cv2.aruco as aruco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R
from rtde_receive import RTDEReceiveInterface
from rtde_control import RTDEControlInterface
from scipy.spatial.transform import Slerp
import time


# ======================================================
# 1 工具函数：加载矩阵
# ======================================================
def load_yaml_T(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return np.array(data['T_EC']) if 'T_EC' in data else np.array(data['T'])


def load_npy_T(path):
    return np.load(path)


# ======================================================
#  加载标定参数与手眼矩阵
# ======================================================
data = np.load("charuco_calibration.npz")
K, dist = data["K"], data["dist"]
print("已加载相机标定参数")

T_E_C = load_yaml_T("T_EC.yaml")     # {}^{E}T_C
T_Cd_G = load_npy_T("T_C*G.npy")    # {}^{C_d}T_G
T_Ed_Cd = T_E_C.copy()               # {}^{E_d}T_{C_d}
print(" 已加载 T_E_C, T_Cd_G, T_Ed_Cd")


# ======================================================
#  初始化 RTDE（接收 + 控制）
# ======================================================
rtde_r = RTDEReceiveInterface("172.17.21.11")
rtde_c = RTDEControlInterface("172.17.21.11")


# ======================================================
#  相机与 ArUco 初始化
# ======================================================
marker_length = 0.03  # Marker 实际边长（米）
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, parameters)

cap = cv2.VideoCapture("/dev/video6")
if not cap.isOpened():
    raise RuntimeError(" 无法打开相机")

SHOW_CAMERA = True  # 👈 是否显示摄像头窗口


# ======================================================
#  5辅助函数定义
# ======================================================
def pose_to_T(pose):
    """UR5的 pose=[x,y,z,Rx,Ry,Rz] 转为 4x4 矩阵"""
    R_mat, _ = cv2.Rodrigues(np.array(pose[3:]))
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = pose[:3]
    return T


def T_to_pose(T):
    """4x4 矩阵转回 UR 控制格式 [x,y,z,Rx,Ry,Rz]"""
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    return np.concatenate([T[:3, 3], rvec.flatten()])


def interpolate_T(T, Kp):
    """对误差矩阵进行平移+旋转插值"""
    R_full = R.from_matrix(T[:3, :3])
    t_full = T[:3, 3]

    # 平移插值
    t_interp = Kp * t_full

    # 旋转插值（使用 Slerp 类）
    q_id = R.from_quat([0, 0, 0, 1])  # 单位旋转
    key_times = [0, 1]
    key_rots = R.from_quat([[0, 0, 0, 1], R_full.as_quat()])
    slerp = Slerp(key_times, key_rots)
    R_interp = slerp([Kp])[0].as_matrix()

    # 构造结果矩阵
    T_interp = np.eye(4)
    T_interp[:3, :3] = R_interp
    T_interp[:3, 3] = t_interp
    return T_interp


# ======================================================
# 主循环：检测 ArUco + 计算 + 控制
# ======================================================
Kp = 0.05  # 插值比例（步长控制）
print(" 开始实时视觉伺服控制... 按 Q 退出")

T_C_G_prev = None
alpha = 0.15  # 滤波系数（越小越平滑）

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, K, dist)

        for rvec, tvec, id_ in zip(rvecs, tvecs, ids):
            # === 1️⃣ 目标相对于相机 ===
            R_C_G, _ = cv2.Rodrigues(rvec)
            T_C_G = np.eye(4)
            T_C_G[:3, :3] = R_C_G
            T_C_G[:3, 3] = tvec.flatten()

            # === 滤波: 平滑当前检测结果 ===
            if T_C_G_prev is not None:
                # 平移部分滑动平均
                T_C_G[:3, 3] = alpha * T_C_G[:3, 3] + (1 - alpha) * T_C_G_prev[:3, 3]

                # 旋转部分用四元数球面插值
                R_prev = R.from_matrix(T_C_G_prev[:3, :3])
                R_curr = R.from_matrix(T_C_G[:3, :3])
                slerp = Slerp([0, 1], R.from_matrix([R_prev.as_matrix(), R_curr.as_matrix()]))
                R_interp = slerp([alpha])[0].as_matrix()
                T_C_G[:3, :3] = R_interp

            # 保存当前矩阵用于下一帧
            T_C_G_prev = T_C_G.copy()


            # === 2️⃣ 当前误差矩阵 {}^{E}T_{E_d} ===
            T_E_Ed = (T_E_C @ T_C_G) @ np.linalg.inv(T_Ed_Cd @ T_Cd_G)
            T_E_Ed_interp = interpolate_T(T_E_Ed, Kp)

            # === 3️⃣ 当前末端姿态 ===
            current_pose = rtde_r.getActualTCPPose()
            T_current = pose_to_T(current_pose)

            # === 4️⃣ 计算目标姿态 ===
            T_target = T_current @ T_E_Ed_interp
            pose_target = T_to_pose(T_target)

            # === 5️⃣ 输出调试信息 ===
            print(f"\n🔹 检测到 ArUco ID {id_}")
            print("当前误差矩阵 {}^{E}T_{E_d} =\n", np.round(T_E_Ed, 4))
            print("插值后 {}^{E}T_{E_d}^{interp} =\n", np.round(T_E_Ed_interp, 4))
            print("目标末端姿态 [x,y,z,Rx,Ry,Rz] =\n", np.round(pose_target, 4))

            # === 6️⃣ 控制 UR5（建议初次运行注释掉） ===
            rtde_c.servoL(pose_target, 0.05, 0.02, 0.008, 0.15, 200)

            # 让下一次检测稍微等一下（否则太频繁）
            time.sleep(1/30)
            

            # === 7️⃣ 可视化坐标轴 ===
            if SHOW_CAMERA:
                cv2.drawFrameAxes(frame, K, dist, rvec, tvec, 0.03)

    # ===  显示摄像头窗口（可开关） ===
    if SHOW_CAMERA:
        cv2.imshow("Aruco Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

# =====================================================
# 退出清理
# ======================================================
cap.release()
cv2.destroyAllWindows()
rtde_c.servoStop()
print("✅ 程序已退出并关闭连接。")
