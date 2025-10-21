import cv2
import cv2.aruco as aruco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R


# ======================================================
# 1  工具函数：加载矩阵
# ======================================================
def load_yaml_T(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return np.array(data['T_EC']) if 'T_EC' in data else np.array(data['T'])


def load_npy_T(path):
    return np.load(path)


# ======================================================
# 2  加载标定参数和手眼标定矩阵
# ======================================================
data = np.load("charuco_calibration.npz")
K, dist = data["K"], data["dist"]
print(" 已加载相机标定参数")

T_E_C = load_yaml_T("T_EC.yaml")     # {}^{E}T_C
T_Cd_G = load_npy_T("T_Cd*G.npy")    # {}^{C_d}T_G
T_Ed_Cd = T_E_C.copy()               # 假设理想状态相同
print(" 已加载 T_E_C, T_Cd_G, T_Ed_Cd")


# ======================================================
# 3  打开相机并检测 ArUco（实时）
# ======================================================
marker_length = 0.03
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, parameters)

cap = cv2.VideoCapture("/dev/video6")
if not cap.isOpened():
    raise RuntimeError(" 无法打开相机")


# ======================================================
# 4 定义插值函数
# ======================================================
def interpolate_T(T, Kp):
    R_full = R.from_matrix(T[:3, :3])
    t_full = T[:3, 3]

    # 平移插值
    t_interp = Kp * t_full

    # 旋转插值
    q_full = R_full.as_quat()
    q_id = np.array([0, 0, 0, 1])
    r_interp = R.slerp(0, 1, [R.from_quat(q_id), R.from_quat(q_full)])([Kp])[0]
    R_interp = r_interp.as_matrix()

    T_interp = np.eye(4)
    T_interp[:3, :3] = R_interp
    T_interp[:3, 3] = t_interp
    return T_interp


# ======================================================
# 5  主循环：检测并计算 {}^{E}T_{E_d}
# ======================================================
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
            R_C_G, _ = cv2.Rodrigues(rvec)
            T_C_G = np.eye(4)
            T_C_G[:3, :3] = R_C_G
            T_C_G[:3, 3] = tvec.flatten()

            # === 计算 {}^{E}T_{E_d} ===
            T_E_Ed = (T_E_C @ T_C_G) @ np.linalg.inv(T_Ed_Cd @ T_Cd_G)

            # === 插值 ===
            Kp = 0.3  # 比例系数（可调节）
            T_E_Ed_interp = interpolate_T(T_E_Ed, Kp)

            print(f"\n🔹 检测到 ArUco ID {id_}")
            print("当前 {}^{E}T_{E_d} =\n", np.round(T_E_Ed, 4))
            print("插值后目标矩阵 T_E_Ed_interp(Kp=0.3) =\n", np.round(T_E_Ed_interp, 4))

            # 可视化坐标轴
            cv2.drawFrameAxes(frame, K, dist, rvec, tvec, 0.03)

    cv2.imshow("Aruco Detection", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
