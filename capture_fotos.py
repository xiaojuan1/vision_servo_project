import cv2
import os

# === 1. 设置保存路径 ===
save_dir = "charuco_images"
os.makedirs(save_dir, exist_ok=True)

# === 2. 打开摄像头 ===
cap = cv2.VideoCapture("/dev/video6") # 
if not cap.isOpened():
    print("❌ 摄像头未打开，请检查编号！")
    exit()

print("✅ 摄像头已启动，按空格拍照，按 ESC 退出。")

i = 1
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 无法读取图像帧。")
        break

    # 显示预览画面
    cv2.imshow("Camera Preview", frame)

    key = cv2.waitKey(1) & 0xFF

    # === 空格键拍照 ===
    if key == 32:  # 空格键
        filename = os.path.join(save_dir, f"charuco_{i:02d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"📸 已保存: {filename}")
        i += 1

    # === ESC 键退出 ===
    elif key == 27:  # ESC
        print("👋 退出拍照模式。")
        break

cap.release()
cv2.destroyAllWindows()
