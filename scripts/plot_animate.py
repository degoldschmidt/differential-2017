"""
=================
An animated image
=================

This example demonstrates how to animate an image.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()


def f(x, y):
    return np.sin(x) + np.cos(y)

x = np.linspace(0, 2 * np.pi, 120)
y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

random_stack = np.random.random((1000,50,50))
im = plt.imshow(random_stack[0], animated=True)


def updatefig(i):
    im.set_array(random_stack[i])
    return im,

ani = animation.FuncAnimation(fig, updatefig, np.arange(1000), interval=50, blit=True)
plt.show()
