import numpy as np
from utils import visualize_solution, get_exact_solution
import matplotlib.pyplot as plt

U_frames = get_exact_solution(ht=0.00005, T=0.5)[::50, :, :]
U_frames_scheme = np.load('U_scheme1_explicit.npy')

values = []

for t in range(U_frames.shape[0]):
    diff = np.abs(U_frames[t] - U_frames_scheme[t])

    diff = np.sum(diff)

    values.append(diff)

plt.plot(range(U_frames.shape[0]), values)
# plt.yscale('log')
plt.xlabel('Time step')
plt.ylabel('L1 error')
plt.savefig('error.png')

# t = 10
# plt.imshow(np.abs(U_frames[t] - U_frames_scheme[t]), cmap='hot', interpolation='nearest')
# plt.colorbar()
# plt.savefig('diff_t100.png')

# visualize_solution(U_frames_scheme, fps=800)
# visualize_solution(np.abs(U_frames - U_frames_scheme), fps=500)
