import numpy as np
T_new = np.loadtxt("T_C*G.txt")
np.save("T_C*G.npy", T_new)
print("✅ 已将修改后的矩阵重新保存为 T_C*G.npy")
