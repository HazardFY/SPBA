import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
def visualize_two_point_clouds(points1, points2, color1=[1, 0, 0], color2=[0.7, 0.7, 0.7]):
    
    if points1.shape[1] != 3 or points2.shape[1] != 3:
        raise ValueError("wrong shape")



    point_cloud1 = o3d.geometry.PointCloud()
    point_cloud1.points = o3d.utility.Vector3dVector(points1)
    point_cloud1.paint_uniform_color(color1)


    point_cloud2 = o3d.geometry.PointCloud()
    point_cloud2.points = o3d.utility.Vector3dVector(points2)
    point_cloud2.paint_uniform_color(color2)


    vis = o3d.visualization.Visualizer()
    vis.create_window()


    vis.add_geometry(point_cloud1)
    vis.add_geometry(point_cloud2)

    opt = vis.get_render_option()
    opt.point_size = 5.0  


    ctr = vis.get_view_control()
    ctr.set_front([0, -1, 0])  
    ctr.set_lookat([0.5, 0.5, 0.5])  
    ctr.set_up([0, 0, -1])  
    ctr.set_zoom(0.8)  


    vis.run()
    vis.destroy_window()

def plot_frequency_spectrum_bar(freq_points):
    """

    """

    if freq_points.is_cuda:
        freq_points = freq_points.detach().cpu().numpy()
    else:
        freq_points = freq_points.detach().numpy()

    frequencies = freq_points 

    x_freq = frequencies[0]  # [N]
    y_freq = frequencies[1]  # [N]
    z_freq = frequencies[2]  # [N]

    freq_indices = np.arange(len(x_freq))

    bar_width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(freq_indices - bar_width, x_freq, width=bar_width, color='red', alpha=0.7, label='X-axis')
    plt.bar(freq_indices, y_freq, width=bar_width, color='green', alpha=0.7, label='Y-axis')
    plt.bar(freq_indices + bar_width, z_freq, width=bar_width, color='blue', alpha=0.7, label='Z-axis')

    plt.axhline(y=0, color='black', linestyle='--')  
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.title("Frequency Spectrum of Point Cloud (Bar Chart)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()