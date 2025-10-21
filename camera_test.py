import cv2

cap = cv2.VideoCapture("/dev/video6")  # 使用 UGREEN 相机主视频流
if not cap.isOpened():
    print("无法打开相机，请检查编号或权限！")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取画面！")
        break

    cv2.imshow("UGREEN Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
