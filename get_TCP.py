from rtde_receive import RTDEReceiveInterface
rtde_r = RTDEReceiveInterface("172.17.21.11") 
print(rtde_r.getActualTCPPose())
