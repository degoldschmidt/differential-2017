import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns

def get_positions(start, radius, each=60):
    ang = np.arange(start,360,each)
    pos = np.zeros((2,len(ang)))
    pos[0,:] = radius * np.cos(np.radians(ang))
    pos[1,:] = radius * np.sin(np.radians(ang))
    return pos, ang

inner_pos, inner_ang = get_positions(30, 1)
outer_pos, outer_ang = get_positions(0, 2)
#outer_pos, outer_ang = get_positions(15, 2, each=15)

print(inner_pos)
print(outer_pos)

colors = ['#ffc04c', '#4c8bff']

f, ax = plt.subplots(figsize=(3,3))

_radius = 1.6
print('Radius: {}'.format(_radius))
for i in range(inner_pos.shape[1]):
    if i==1:
        ax.plot(inner_pos[0,i], inner_pos[1,i], 'r.')
        circ1 = plt.Circle(inner_pos[:,i], radius=_radius, ls='--', fill=False)
        ax.add_artist(circ1)
    circ = plt.Circle(inner_pos[:,i], radius=.15, color=colors[(i+1)%2])
    ax.add_artist(circ)
for i in range(outer_pos.shape[1]):
    if i==1:
        ax.plot(outer_pos[0,i], outer_pos[1,i], 'b.')
        circ2 = plt.Circle(outer_pos[:,i], radius=_radius, ls='--', fill=False)
        ax.add_artist(circ2)
    circ = plt.Circle(outer_pos[:,i], radius=.15, color=colors[(i+1)%2])
    ax.add_artist(circ)

ax.set_xlim([-3,3])
ax.set_ylim([-3,3])
ax.set_aspect('equal')
plt.show()
