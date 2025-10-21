import cv2
from cv2 import aruco
import matplotlib.pyplot as plt

# === 1. 定义 Charuco 棋盘参数 ===
squares_x = 5    # 棋盘格在 x 方向的格子数
squares_y = 7    # 棋盘格在 y 方向的格子数
square_length = 0.03  # 每个棋盘格边长（米）
marker_length = 0.02  # 每个 ArUco 的边长（米）

# === 2. 使用 4x4_50 字典 ===
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# === 3. 创建 Charuco 棋盘 ===
board = aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, dictionary)

# === 4. 生成图像（像素分辨率仅影响清晰度）===
img = board.generateImage((1000, 1400))

# === 5. 保存为 A4 尺寸 PDF（210×297 mm）===
plt.figure(figsize=(8.27, 11.69))  # A4 尺寸（英寸）
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.savefig("charuco_board_A4.pdf", bbox_inches='tight', pad_inches=0)
print("✅ Saved as charuco_board_A4.pdf (A4 actual size)")
