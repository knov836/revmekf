import matplotlib.pyplot as plt
import numpy as np

# --- Rotation matrices ---
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

# --- Angles ---
tilt_angle = np.radians(45)   # plane inclination (45 deg)
z_rotation = np.radians(30)   # rotation around IMU's local Z (30 deg)

# --- Orientations ---
R_tilt = rot_y(tilt_angle)           # IMU orientation when sitting on the tilted plane
R_before = R_tilt                    # before local Z rotation
# IMPORTANT: post-multiply to rotate the IMU around its *local* z-axis
R_after = R_before @ rot_z(z_rotation)  

# --- Sanity check: the Z-axis (3rd column) should remain the same ---
z_before = R_before[:, 2]
z_after  = R_after[:, 2]
print("Z before:", z_before)
print("Z after: ", z_after)
print("Difference norm:", np.linalg.norm(z_before - z_after))
print("All close (within 1e-12):", np.allclose(z_before, z_after, atol=1e-12))

# --- Generate finite inclined plane (tilted around Y) ---
plane_size = 1.0
xx, yy = np.meshgrid(np.linspace(-plane_size, plane_size, 10),
                     np.linspace(-plane_size, plane_size, 10))
zz = np.zeros_like(xx)
points = np.vstack((xx.flatten(), yy.flatten(), zz.flatten()))
rot_points = R_tilt @ points
xx_tilt = rot_points[0, :].reshape(xx.shape)
yy_tilt = rot_points[1, :].reshape(xx.shape)
zz_tilt = rot_points[2, :].reshape(zz.shape)

# --- Plot ---
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

# plane
ax.plot_surface(xx_tilt, yy_tilt, zz_tilt, alpha=0.45, color='lightgray')

# axes origins
origin = np.array([0.0, 0.0, 0.0])

# IMU axes before (dashed)
imu_axes_before = R_before @ np.eye(3)
ax.quiver(*origin, *imu_axes_before[:,0], length=0.6, linewidth=2, linestyle='--', color='tab:orange')
ax.quiver(*origin, *imu_axes_before[:,1], length=0.6, linewidth=2, linestyle='--', color='tab:orange')
ax.quiver(*origin, *imu_axes_before[:,2], length=0.6, linewidth=2, linestyle='--', color='tab:orange')

# IMU axes after (solid)
imu_axes_after = R_after @ np.eye(3)
ax.quiver(*origin, *imu_axes_after[:,0], length=0.6, linewidth=2, linestyle='-', color='r')
ax.quiver(*origin, *imu_axes_after[:,1], length=0.6, linewidth=2, linestyle='-', color='g')
ax.quiver(*origin, *imu_axes_after[:,2], length=0.6, linewidth=2, linestyle='-', color='b')

# world refs (black)
ax.quiver(0,0,0, 1,0,0, length=0.6, linewidth=1.5, color='k')
ax.quiver(0,0,0, 0,1,0, length=0.6, linewidth=1.5, color='k')
ax.quiver(0,0,0, 0,0,1, length=0.6, linewidth=1.5, color='k')
ax.text(1.05*0.6, 0, 0, 'X_ref', color='k')
ax.text(0, 1.05*0.6, 0, 'Y_ref', color='k')
ax.text(0, 0, 1.05*0.6, 'Z_ref', color='k')

# Labels for IMU axes (slightly offset to avoid overlap)
offset = 1.05
ax.text(*(offset * imu_axes_before[:,0]), 'Before X', color='tab:orange')
ax.text(*(offset * imu_axes_before[:,1]), 'Before Y', color='tab:orange')
ax.text(*(offset * imu_axes_before[:,2]), 'Before Z', color='tab:orange')

ax.text(*(offset * imu_axes_after[:,0]), 'After X', color='r')
ax.text(*(offset * imu_axes_after[:,1]), 'After Y', color='g')
ax.text(*(offset * imu_axes_after[:,2]), 'After Z', color='b')

# view and limits
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-0.6, 1.0])
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_title("IMU on finite plane (45° tilt): before (dashed) & after 30° around local Z (solid)")
ax.view_init(elev=25, azim=30)
plt.tight_layout()
plt.show()
