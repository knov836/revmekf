import numpy as np
import matplotlib.pyplot as plt

def visualize_rotations_about_x_centered(A=np.array([0, 0, 1]),
                                         normal=np.array([0, 0, 1]),
                                         center=np.array([0, 0, 0]),
                                         offset=0.5,
                                         angle_range=np.linspace(0, 2*np.pi, 200)):
    """
    Visualize the trajectory of vector A, anchored at base_point = center - A + offset*normal,
    rotated around the X-axis, and show its intersection with the plane (normal, center).
    """
    A = np.array(A, dtype=float)
    normal = np.array(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    center = np.array(center, dtype=float)

    # Base point of vector A: offset along normal
    base_point = center - A + offset * normal

    # Rotation matrix around X-axis
    def rot_x(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[1, 0, 0],
                         [0, c, -s],
                         [0, s, c]])

    # Generate rotated vectors
    rotated_As = np.array([rot_x(theta) @ A for theta in angle_range])
    rotated_tips = base_point + rotated_As  # positions of vector tips

    # Compute intersection with plane
    distances = (rotated_tips - center) @ normal
    crossings = np.where(np.sign(distances[:-1]) * np.sign(distances[1:]) < 0)[0]

    intersections = []
    for i in crossings:
        t = distances[i] / (distances[i] - distances[i + 1])
        inter = rotated_tips[i] + t * (rotated_tips[i + 1] - rotated_tips[i])
        intersections.append(inter)
    intersections = np.array(intersections)

    # --- Plot ---
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plane
    d = -center @ normal
    xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 10), np.linspace(-1.5, 1.5, 10))
    zz = (-normal[0]*xx - normal[1]*yy - d) / normal[2]
    ax.plot_surface(xx, yy, zz, alpha=0.3, color='gray')

    # Plane center and normal
    ax.scatter(*center, color='r', s=60, label='Center C')
    ax.quiver(*center, *normal, color='g', length=0.3, label='Normal n')

    # Vector A (anchored at base_point)
    ax.quiver(*base_point, *A, color='blue', linewidth=2, label='Vector A')

    # Trajectory of rotated vector tips
    ax.plot(rotated_tips[:, 0], rotated_tips[:, 1], rotated_tips[:, 2],
            color='orange', linewidth=2, label='Tip path (rotation around X)')

    # Intersections
    if intersections.size > 0:
        ax.scatter(intersections[:, 0], intersections[:, 1], intersections[:, 2],
                   color='k', s=50, label='Intersections')

    # Axes labels and view
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(loc='upper left')
    ax.set_title("Rotations of A (anchored at C + A - g*dt) around X-axis and intersection with plane")
    ax.view_init(elev=25, azim=45)

    plt.show()

    return rotated_tips, intersections

visualize_rotations_about_x_centered(
    A=[0, 0, 1],          # initial vector
    normal=[0, 0, 1],     # plane normal
    center=[0, 0, 0.5]    # plane center offset along Z
)