from matplotlib import pyplot as plt
import pandas as pd

data = pd.read_excel('./results/data.xlsx')

step = []
for i in range(26):
    step.append(i)

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

ax1.plot(step, data.Ours_CC_5, 'b-', label='Ours+CC (5-shot)')
ax1.plot(step, data.CC_CC_5, 'r-', label='CC+CC (5-shot)')
ax1.plot(step, data.Ours_CC_1, 'b--', label='Ours+CC (1-shot)')
ax1.plot(step, data.CC_CC_1, 'r--', label='CC+CC (1-shot)')

ax2.plot(step, data.Ours_Ours_5, 'b-', label='Ours+Ours (5-shot)')
ax2.plot(step, data.CC_Ours_5, 'r-', label='CC+Ours (5-shot)')
ax2.plot(step, data.Ours_Ours_1, 'b--', label='Ours+Ours (1-shot)')
ax2.plot(step, data.CC_Ours_1, 'r--', label='CC+Ours (1-shot)')

ax1.grid()
ax1.legend()
ax1.set_ylabel('Accuracy (%)')
ax1.set_xlabel('Gradient steps')

ax2.grid()
ax2.legend()
ax2.set_xlabel('Gradient steps')

plt.show()

