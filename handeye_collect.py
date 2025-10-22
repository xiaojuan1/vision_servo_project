import cv2, numpy as np
import cv2.aruco as aruco
from rtde_receive import RTDEReceiveInterface
import pyrealsense2 as rs

# === 初始化 RTDE ===
rtde_r = RTDEReceiveInterface("172.17.21.11")

R_gripper2base, t_gripper2base = [], []
R_target2cam, t_target2cam = [], []

# === 定义 ArUco 检测函数（持续显示画面，按回车拍照）===
def detect_aruco(device="/dev/video6", marker_length=0.03, show=True):
    """持续显示相机画面，按回车采集当前帧，返回 rvec, tvec"""
    # === 加载相机内参 ===
    data = np.load("charuco_calibration.npz")
    K, dist = data["K"], data["dist"]

    # === 打开相机 ===
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(" 无法打开相机，请检查设备号或权限")

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, parameters)

    print("📷 实时画面已开启，请将 ArUco 放入视野。按 Enter 拍照采集该帧。")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("相机帧获取失败")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, K, dist)
            for rvec, tvec in zip(rvecs, tvecs):
                cv2.drawFrameAxes(frame, K, dist, rvec, tvec, 0.03)
        else:
            rvecs, tvecs = None, None

        # 显示画面
        if show:
            cv2.imshow("Aruco Detection (Live)", frame)

        # 检查键盘输入
        key = cv2.waitKey(1)
        if key == 13:  # Enter 键的 ASCII 码是 13
            break
        elif key == 27:  # ESC 退出
            cap.release()
            cv2.destroyAllWindows()
            raise RuntimeError("手动退出采集。")

    cap.release()
    cv2.destroyAllWindows()

    if ids is None:
        raise RuntimeError(" 当前帧未检测到 ArUco，请重试。")
    return rvecs[0], tvecs[0]


# === 主循环：采集 10 组数据 ===
for i in range(10):
    input(f"\n👉 请移动机械臂到第 {i+1} 个位置后按回车继续...")

    # === 从 RTDE 获取末端姿态 ===
    pose = rtde_r.getActualTCPPose()
    x, y, z, Rx, Ry, Rz = pose
    R_BE, _ = cv2.Rodrigues(np.array([Rx, Ry, Rz]))
    t_BE = np.array([[x], [y], [z]])
    R_gripper2base.append(R_BE)
    t_gripper2base.append(t_BE)

    # === 从相机检测 ArUco（实时显示 + 回车采集）===
    try:
        rvec, tvec = detect_aruco(show=True)
        R_CG, _ = cv2.Rodrigues(rvec)
        t_CG = tvec.reshape(3,1)
        R_target2cam.append(R_CG)
        t_target2cam.append(t_CG)
        print(f"✅ 第 {i+1} 组数据保存完成！")
    except Exception as e:
        print(f"第 {i+1} 组数据失败：{e}")
        continue

# === 关闭所有窗口 ===
cv2.destroyAllWindows()

# === 保存数据 ===
np.savez("handeye_data.npz",
         R_gripper2base=R_gripper2base, t_gripper2base=t_gripper2base,
         R_target2cam=R_target2cam, t_target2cam=t_target2cam)

print("\n✅ 所有数据已保存到 handeye_data.npz")
