import cv2
from cv2 import aruco
import numpy as np
import glob

# === 1. 定义 Charuco 棋盘参数 ===
squares_x = 5
squares_y = 7
square_length = 0.03
marker_length = 0.02
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
board = aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, dictionary)

# === 2. 读取拍摄的标定图像 ===
images = glob.glob('charuco_images/*.jpg')
all_corners = []
all_ids = []
img_size = None

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_size = gray.shape[::-1]

    # 检测 ArUco 标记
    corners, ids, rejected = aruco.detectMarkers(gray, dictionary)

    if len(corners) > 0:
        # 优化检测
        aruco.refineDetectedMarkers(gray, board, corners, ids, rejected)
        # 检测 Charuco 角点
        retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            corners, ids, gray, board
        )
        if retval > 10:  # 至少检测到10个角点才进行标定
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)

# === 3. 标定 ===
ret, K, dist, rvecs, tvecs = aruco.calibrateCameraCharuco(
    all_corners, all_ids, board, img_size, None, None
)

# === 4. 输出结果 ===
print("✅ 相机标定完成！")
print("内参矩阵 K：\n", K)
print("畸变系数 dist：\n", dist.ravel())

# === 5. 保存结果 ===
np.savez("charuco_calibration.npz", K=K, dist=dist)
print("已保存到 charuco_calibration.npz")
