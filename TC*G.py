import cv2
import cv2.aruco as aruco
import numpy as np
import os

# === 1. 读取标定文件 ===
data = np.load("charuco_calibration.npz")
K = data["K"]
dist = data["dist"]

print("✅ 已加载标定参数")
print("K =\n", K)
print("dist =", dist.ravel())

# === 2. 打开相机 ===
cap = cv2.VideoCapture("/dev/video6")
if not cap.isOpened():
    print("相机无法打开！请检查设备号。")
    exit()

# === 3. 定义 ArUco 字典与检测器参数 ===
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, parameters)

# === 4. 设定实际 marker 尺寸（单位：米）===
marker_length = 0.03  # 3 cm

# === 5. 保存路径 ===
save_dir = "aruco_poses"
os.makedirs(save_dir, exist_ok=True)

def get_T_camera_target(rvec, tvec):
    """将 rvec, tvec 转为 4x4 齐次变换矩阵"""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # === 检测 ArUco ===
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)

        # === 位姿估计 ===
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, K, dist)
        for rvec, tvec, id_ in zip(rvecs, tvecs, ids):
            cv2.drawFrameAxes(frame, K, dist, rvec, tvec, 0.03)

            # 生成齐次矩阵
            T = get_T_camera_target(rvec, tvec)

            # 打印输出
            print(f"\n🟩 Marker ID {id_[0]}")
            print("tvec (m):", np.round(tvec.flatten(), 4))
            print("rvec (rad):", np.round(rvec.flatten(), 4))
            print("T_camera_target =\n", np.round(T, 4))

            # === 保存矩阵 ===
            filename = os.path.join(save_dir, f"marker_{id_[0]}_frame_{frame_id}.npy")
            np.save(filename, T)

            # 或保存为文本文件：
            txt_filename = filename.replace(".npy", ".txt")
            np.savetxt(txt_filename, T, fmt="%.6f")

        frame_id += 1

    cv2.imshow("ArUco Detection (Calibrated)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n✅ 已保存所有变换矩阵到文件夹： {save_dir}")
