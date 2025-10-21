from rtde_receive import RTDEReceiveInterface
rtde_r = RTDEReceiveInterface("172.17.21.11")  # 替换成你的UR5 IP
print(rtde_r.getActualTCPPose())
