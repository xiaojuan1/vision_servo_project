import cv2
import cv2.aruco as aruco
import numpy as np

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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # === 5. 检测 ArUco ===
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        # 绘制检测框
        aruco.drawDetectedMarkers(frame, corners, ids)

        # === 6. 位姿估计 ===
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, K, dist)
        for rvec, tvec, id_ in zip(rvecs, tvecs, ids):
            cv2.drawFrameAxes(frame, K, dist, rvec, tvec, 0.03)

            # 输出位姿信息
            
            print("tvec (m):", np.round(tvec.flatten(), 4))
            print("rvec (rad):", np.round(rvec.flatten(), 4))

    cv2.imshow("ArUco Detection (Calibrated)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
