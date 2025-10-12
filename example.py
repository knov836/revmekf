import matplotlib.pyplot as plt
import numpy as np

# --- Rotation Matrices ---
def rot_y(theta):
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

def rot_z(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])

# --- Define plane tilt and IMU rotation ---
tilt_angle = np.radians(45)   # Plane inclined 45° from horizontal
z_rotation = np.radians(30)   # IMU rotated 30° around its local Z-axis

# Rotation of the plane (around Y-axis)
R_tilt = rot_y(tilt_angle)
# IMU orientation (tilt + rotation around its Z)
R_imu = rot_z(z_rotation) @ R_tilt

# --- Generate finite inclined plane ---
plane_size = 1.0
xx, yy = np.meshgrid(np.linspace(-plane_size, plane_size, 10),
                     np.linspace(-plane_size, plane_size, 10))
zz = np.zeros_like(xx)

# Rotate the plane points
points = np.vstack((xx.flatten(), yy.flatten(), zz.flatten()))
rot_points = R_tilt @ points
xx_tilt = rot_points[0, :].reshape(xx.shape)
yy_tilt = rot_points[1, :].reshape(yy.shape)
zz_tilt = rot_points[2, :].reshape(zz.shape)

# --- Create 3D figure ---
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the inclined plane
ax.plot_surface(xx_tilt, yy_tilt, zz_tilt, alpha=0.5, color='lightgray')

# --- Plot IMU axes ---
origin = np.array([0, 0, 0])
imu_axes = R_imu @ np.eye(3)
ax.quiver(*origin, *imu_axes[:,0], color='r', linewidth=3, linestyle='--')
ax.quiver(*origin, *imu_axes[:,1], color='g', linewidth=3, linestyle='--')
ax.quiver(*origin, *imu_axes[:,2], color='b', linewidth=3, linestyle='--')

# --- Reference axes (world frame) ---
ax.quiver(0, 0, 0, 1, 0, 0, color='r', linewidth=2)
ax.quiver(0, 0, 0, 0, 1, 0, color='g', linewidth=2)
ax.quiver(0, 0, 0, 0, 0, 1, color='b', linewidth=2)

# --- Labels ---
ax.text(1.1, 0, 0, 'X_ref', color='r')
ax.text(0, 1.1, 0, 'Y_ref', color='g')
ax.text(0, 0, 1.1, 'Z_ref', color='b')
ax.text(*imu_axes[:,0], 'X_IMU', color='r')
ax.text(*imu_axes[:,1], 'Y_IMU', color='g')
ax.text(*imu_axes[:,2], 'Z_IMU', color='b')

# --- Plot settings ---
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-0.5, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("IMU on an inclined plane (45° tilt, 30° Z-rotation)")
ax.view_init(elev=25, azim=35)

plt.tight_layout()
plt.show()
