import cv2
import cv2.aruco as aruco

# 创建一个字典（4x4_50）
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# 生成一个ID为0的标记
marker = aruco.generateImageMarker(dictionary, 0, 400)

# 保存为图片
cv2.imwrite("aruco_id0.png", marker)
cv2.imshow("ArUco Marker", marker)
cv2.waitKey(0)
cv2.destroyAllWindows()
