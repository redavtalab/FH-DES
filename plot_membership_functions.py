from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.patches as patches

def mf_classification(a1,a2,v,w):
  gamma = 0.95
  b =     np.maximum(0,(1-np.maximum(0,gamma*np.minimum(1,a1-w[0])))) + np.maximum(0,1-np.maximum(0,gamma*np.minimum(1,v[0]-a1)))
  b = b + np.maximum(0,(1-np.maximum(0,gamma*np.minimum(1,a2-w[1])))) + np.maximum(0,1-np.maximum(0,gamma*np.minimum(1,v[1]-a2)))
  b = (1/(2*2)) * b
  return b
def f(r,y):
  x = r * y
  x[x * y > 1] = 1
  x[x * y < 0] = 0
  return x*y
def mf_clustering(a1,a2,v,w):
  gamma = 0.5
  b =     1- f(a1-w[0],gamma) - f(v[0]-a1,gamma)
  b = b + 1- f(a2-w[1],gamma) - f(v[1]-a2,gamma)
  b = (1/2) * b
  return b
def mf_gabrys(a0,a1,v,w):
  boxes_center = ((w + v) / 2)
  halfsize = ((w - v) / 2)
  d0 = np.abs(boxes_center[0] - a0) - halfsize[0]
  d1 = np.abs(boxes_center[1] - a1) - halfsize[1]
  d0[d0 < 0] = 0
  d1[d1 < 0] = 0

  dd = np.maximum(d0,d1)
  dd = dd / 2 # number of dimensions
  m = 1 - dd  # m: membership
  m = np.power(m, 4)
  return m
def mf_proposed(a0,a1,v,w):
  boxes_center = ((w + v) / 2)
  halfsize = ((w - v) / 2)
  d0 = np.abs(boxes_center[0] - a0) - halfsize[0]
  d1 = np.abs(boxes_center[1] - a1) - halfsize[1]
  d0[d0 < 0] = 0
  d1[d1 < 0] = 0

  dd = np.linalg.norm(np.array([d0,d1]), axis=0)
  dd = dd / 2 # number of dimensions
  m = 1 - dd  # m: membership
  m = np.power(m, 4)
  return m
def hb_rect(a0,a1):
  z = np.ones_like(a0)
  return z+0.001

x = np.linspace(0, 1, 1000)
y = np.linspace(0, 1, 1000)
X, Y = np.meshgrid(x, y)

v1 = np.array([0.1 , 0.3])
w1 = np.array([0.3 , 0.6])
Z1 = mf_classification(X,Y,v1,w1)
# Z1 = mf_clustering(X,Y,v1,w1)
# Z1 = mf_gabrys(X,Y,v1,w1)
# Z1 = mf_proposed(X,Y,v1,w1)

xh1 = np.linspace(v1[0],w1[0],10)
yh1 = np.linspace(v1[1],w1[1],10)
Xh1,Yh1 = np.meshgrid(xh1,yh1)
Zh1 = hb_rect(Xh1,Yh1)


v2 = np.array([0.5 , 0.6])
w2 = np.array([0.7 , 0.9])
Z2 = mf_classification(X,Y,v2,w2)
# Z2 = mf_clustering(X,Y,v2,w2)
# Z2 = mf_gabrys(X,Y,v2,w2)
# Z2 = mf_proposed(X,Y,v2,w2)

xh2 = np.linspace(v2[0],w2[0],10)
yh2 = np.linspace(v2[1],w2[1],10)
Xh2,Yh2 = np.meshgrid(xh2,yh2)
Zh2 = hb_rect(Xh2,Yh2)





print("max:",np.max(Z1), "  min:",np.min(Z1))

# Add the patch to the Axes


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# ax.plot_surface(X, Y, Z1, norm=None, vmin=None, vmax=None, facecolors=cm.jet(Z1 * Z1))
# ax.contour(X, Y, Z1, extend3d=False, levels=15, stride=20, zdir='z', offset=None, data=None)

# ax.plot_surface(X, Y, Z2, norm=None, vmin=None, vmax=None, facecolors=cm.jet(Z2 * Z2))
# ax.contour(X, Y, Z2, extend3d=False, levels=15, stride=20, zdir='z', offset=None, data=None)
ax.plot_wireframe(Xh1, Yh1, Zh1, rstride=1, cstride=1)
ax.plot_wireframe(Xh2, Yh2, Zh2, rstride=1, cstride=1)

ax.plot_surface(X, Y, Z2>Z1, norm=None, vmin=None, vmax=None)#, facecolors=cm.jet(Z2 * Z2))
# rotate the axes and update
for angle in range(90, 92):
    ax.view_init( 90, 0) #np.abs(angle/2 - 90)
    plt.draw()
    plt.pause(10)