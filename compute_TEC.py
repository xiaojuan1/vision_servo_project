import numpy as np, cv2

data = np.load("handeye_data.npz", allow_pickle=True)
R_gripper2base = data["R_gripper2base"]
t_gripper2base = data["t_gripper2base"]
R_target2cam = data["R_target2cam"]
t_target2cam = data["t_target2cam"]

R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
    R_gripper2base, t_gripper2base,
    R_target2cam, t_target2cam,
    method=cv2.CALIB_HAND_EYE_TSAI
)

T_EC = np.eye(4)
T_EC[:3,:3] = R_cam2gripper
T_EC[:3,3] = t_cam2gripper.flatten()
print("🔹T_EC =\n", np.round(T_EC, 4))
