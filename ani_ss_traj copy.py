
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import math
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm
from matplotlib.animation import FFMpegWriter


# Define your save location
dir_1 = 'FINALLogs'
dir_2 = 'Lemni_1'
mapping_dir_3 = 'Lemni_1'
directory_path = f"{dir_1}/{dir_2}"
save_location = directory_path + '/' + 'Lemni_1_RMSE' +'.png'
from transforms3d.euler import euler2mat

#End Effector Length
manipulator_length = 125 #[mm]
r_1 = 5 #[mm]
r_2 = 3.5

# 1. Data Loading and Processing
file_dir = f"{directory_path}/{dir_2}.csv"
print(f'Logging File: {file_dir}')

# Read CSV file
data = pd.read_csv(file_dir)

#For Star T = 40 2 periods
#start_time = 415
#end_time = 455

#For Lemni T= 50 2 periods
#
start_time = 441
end_time = 491
#end_time = 490

# Filter data based on start and end time
filtered_data = data[(data['Time'] >= start_time) & (data['Time'] <= end_time)]
filtered_data['Time'] = filtered_data['Time'] - start_time
filtered_data = filtered_data.iloc[::16, :]

# Adjust time data
#filtered_data['Time'] = filtered_data['Time'] - start_time

# Extract data columns
time_data = filtered_data['Time']
ref_angle_phi = filtered_data['Ref_Joint_Pos_1']
ref_angle_theta = filtered_data['Ref_Joint_Pos_2']
true_angle_phi = filtered_data['Joint_Pos_1']
true_angle_theta = filtered_data['Joint_Pos_2']
est_angle_phi = filtered_data['Est_Joint_Pos_1']
est_angle_theta = filtered_data['Est_Joint_Pos_2']
ref_pos_p1 = filtered_data['Hsl_Ref_1']
ref_pos_p2 = filtered_data['Hsl_Ref_2']
ref_pos_t1 = filtered_data['Hsl_Ref_3']
ref_pos_t2 = filtered_data['Hsl_Ref_4']
true_pos_p1 = filtered_data['Hsl_Pos_1']
true_pos_p2 = filtered_data['Hsl_Pos_2']
true_pos_t1 = filtered_data['Hsl_Pos_3']
true_pos_t2 = filtered_data['Hsl_Pos_4']
est_pos_p1 = filtered_data['Est_Hsl_Pos_1']
est_pos_p2 = filtered_data['Est_Hsl_Pos_2']
est_pos_t1 = filtered_data['Est_Hsl_Pos_3']
est_pos_t2 = filtered_data['Est_Hsl_Pos_4']
control_p1 = filtered_data['Control_Input_1']
control_p1 = 2 * control_p1  # Account for TREK 2
control_p2 = filtered_data['Control_Input_2']
control_t1 = filtered_data['Control_Input_3']
control_t2 = filtered_data['Control_Input_4']
voltage_p1 = filtered_data['Hsl_V_1']
voltage_p2 = filtered_data['Hsl_V_2']
voltage_t1 = filtered_data['Hsl_V_3']
voltage_t2 = filtered_data['Hsl_V_4']

ref_angle_phi = np.deg2rad(ref_angle_phi)
ref_angle_theta = np.deg2rad(ref_angle_theta)
est_angle_theta = np.deg2rad(est_angle_theta)
est_angle_phi = np.deg2rad(est_angle_phi)
true_angle_phi = np.deg2rad(true_angle_phi)
true_angle_theta = np.deg2rad(true_angle_theta)


zero_column = np.zeros_like(ref_angle_phi)

eul_ref = np.column_stack((zero_column, ref_angle_theta, ref_angle_phi))
eul_true = np.column_stack((zero_column, true_angle_theta, true_angle_phi))
eul_est = np.column_stack((zero_column, est_angle_theta, est_angle_phi))

pos_manipulator = np.array([0, 0, -manipulator_length])
pos_r1 = np.array([0, 0, -r_1])

end_effector_ref = np.zeros_like(eul_ref)
end_effector_est = np.zeros_like(eul_est)
end_effector_true = np.zeros_like(eul_true)

attachment_ref = np.zeros_like(eul_ref)
attachment_est = np.zeros_like(eul_est)
attachment_true = np.zeros_like(eul_true)

for i in range(eul_ref.shape[0]):
    # Converts Euler angles to rotation matrix (ZYX order as assumed)
    rotm = euler2mat(eul_ref[i, 0], eul_ref[i, 1], eul_ref[i, 2], 'rzyx')
    new_pos = rotm @ pos_manipulator  # Matrix multiplication
    new_pos_r1 = rotm @ pos_r1
    end_effector_ref[i, :] = new_pos
    attachment_ref[i, :] = new_pos_r1


x_ref = end_effector_ref[:, 0]
y_ref = end_effector_ref[:, 1]
z_ref = end_effector_ref[:, 2]

x_ref_attachment = attachment_ref[:, 0]
y_ref_attachment = attachment_ref[:, 1]

for i in range(eul_est.shape[0]):
    rotm = euler2mat(eul_est[i, 0], eul_est[i, 1], eul_est[i, 2], 'rzyx')
    new_pos = rotm @ pos_manipulator  
    new_pos_r1 = rotm @ pos_r1
    end_effector_est[i, :] = new_pos
    attachment_est[i, :] = new_pos_r1


x_est = end_effector_est[:, 0]
y_est = end_effector_est[:, 1]
z_est = end_effector_est[:, 2]

x_est_attachment = attachment_est[:, 0]
y_est_attachment = attachment_est[:, 1]

# For eul_true
for i in range(eul_true.shape[0]):
    rotm = euler2mat(eul_true[i, 0], eul_true[i, 1], eul_true[i, 2], 'rzyx')
    new_pos = rotm @ pos_manipulator  
    new_pos_r1 = rotm @ pos_r1 
    end_effector_true[i, :] = new_pos
    attachment_true[i, :] = new_pos_r1


x_true = end_effector_true[:, 0]
y_true = end_effector_true[:, 1]
z_true = end_effector_true[:, 2]

x_true_attachment = attachment_true[:, 0]
y_true_attachment = attachment_true[:, 1]

def pos_to_hsl_displacement(x0, y0, r2):
    q_phi_1 = [math.sqrt((r2 - xi)**2 + yi**2) for xi, yi in zip(x0, y0)]
    q_phi_2 = [math.sqrt((r2 + xi)**2 + yi**2) for xi, yi in zip(x0, y0)]
    q_theta_1 = [math.sqrt(xi**2 + (r2 - yi)**2) for xi, yi in zip(x0, y0)]
    q_theta_2 = [math.sqrt(xi**2 + (r2 + yi)**2) for xi, yi in zip(x0, y0)]
    
    return q_phi_1, q_phi_2, q_theta_1, q_theta_2

# Example usage:
ref_pos_p1 ,ref_pos_p2, ref_pos_t1 ,ref_pos_t2 = pos_to_hsl_displacement(x_ref_attachment, y_ref_attachment, r_2)
est_pos_p1 ,est_pos_p2, est_pos_t1 ,est_pos_t2 = pos_to_hsl_displacement(x_est_attachment, y_est_attachment, r_2)
true_pos_p1 ,true_pos_p2, true_pos_t1 ,true_pos_t2 = pos_to_hsl_displacement(x_true_attachment, y_true_attachment, r_2)

distances_true_ref = np.sqrt((x_true - x_ref)**2 + (y_true - y_ref)**2 + (z_true - z_ref)**2)

# Calculate distances between true and estimated for all samples
distances_true_est = np.sqrt((x_true - x_est)**2 + (y_true - y_est)**2 + (z_true - z_est)**2)

# Compute RMSE for the entire end-effector position between true and reference
rmse_true_ref = np.sqrt((distances_true_ref**2).mean())

# Compute RMSE for the entire end-effector position between true and estimated
rmse_true_est = np.sqrt((distances_true_est**2).mean())

# Print RMSE values
print("RMSE between true and reference:", rmse_true_ref)
print("RMSE between true and estimated:", rmse_true_est)

# Plot ref, actual, and estimated angles
A = 6  # Want figures to be A6
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig = plt.figure(figsize=(9, 3), facecolor='k')
plt.rcParams['axes.labelsize'] = 8   # Set x and y labels fontsize
plt.rcParams['legend.fontsize'] = 8  # Set legend fontsize
plt.rcParams['xtick.labelsize'] = 6  # Set x tick labels fontsize
plt.rcParams['ytick.labelsize'] = 6  # Set y tick labels fontsize

gs = fig.add_gridspec(2,4, width_ratios=[1, 1, 3, 3])
ax1 = fig.add_subplot(gs[0, 2])
ax2 = fig.add_subplot(gs[1, 2])
ax3 = fig.add_subplot(gs[1, 3])
ax4 = fig.add_subplot(gs[0, 3])
ax5 = fig.add_subplot(gs[:, :2], projection='3d')
ax5.set_proj_type('ortho')  # FOV = 0 deg

est_color='#FF6666'
true_color =  '#33FF33'
t2_color = '#CCCC00'
t1_color = '#00CC66'
p2_color = '#33FFFF'
p1_color = '#66B2FF'

# Plotting X, Y, Z for different datasets
ax5.plot(x_ref, y_ref, z_ref, color=(1, 1,1), linestyle='--', linewidth=2, label='Reference')
ax5.plot(x_est, y_est, z_est, color=est_color, linestyle='--', linewidth=2, label='Estimated',alpha = 0)
ax5.plot(x_true, y_true, z_true, color=true_color, linewidth=2, label='True',alpha = 0)

ax5.set_xlabel(r'X (mm)',color='white')  # Replace units with appropriate units
ax5.set_ylabel(r'Y (mm)',color='white')  # Replace units with appropriate units
ax5.set_zlabel(' ')  # Replace units with appropriate units

# Setting limits assuming you want them to be similar. Adjust if required.

ax5.set_zlim([-manipulator_length-0.5, -manipulator_length+20])
ax5.grid(True)
ax5.view_init(elev=90, azim=0)

#ax5.view_init(elev=90, azim=0)
#ax5.view_init(elev=45, azim=-155)
ax5.xaxis.pane.fill = True
ax5.yaxis.pane.fill = True
ax5.zaxis.pane.fill = True
ax5.xaxis.pane.set_edgecolor('w')
ax5.yaxis.pane.set_edgecolor('w')
ax5.zaxis.pane.set_edgecolor('w')
ax5.xaxis.pane.fill = True
ax5.yaxis.pane.fill = True
ax5.zaxis.pane.fill = True
ax5.xaxis.pane.set_edgecolor('w')
ax5.yaxis.pane.set_edgecolor('w')
ax5.zaxis.pane.set_edgecolor('w')
ax5.xaxis.pane.set_facecolor((0, 0, 0, 0))  # RGBA values (you can adjust the numbers to your liking)
ax5.yaxis.pane.set_facecolor((0, 0, 0, 0))
ax5.zaxis.pane.set_facecolor((0, 0, 0, 0))
ax5.xaxis._axinfo["grid"]['color'] =  (0.6, 0.6, 0.6, 0.6)  # White RGBA
ax5.yaxis._axinfo["grid"]['color'] =  (0.6, 0.6, 0.6, 0.6)   # White RGBA
ax5.zaxis._axinfo["grid"]['color'] =  (1, 1, 1, 1)  # White RGBA
#for star
#ax5.set_xlim([-28,28])
#ax5.set_ylim([-28,28])

#for lemni
ax5.set_xlim([-40,40])
ax5.set_ylim([-40,40])




z_ticks = ax5.get_zticks()
# Skip every second tick
new_z_ticks = z_ticks[::2]
ax5.set_zticks(new_z_ticks)
ax5.set_zticks([])

y1_ticks = ax5.get_yticks()

# Skip every second tick
new_y1_ticks = y1_ticks[::1]

ax5.set_yticks(new_y1_ticks)

x2_ticks = ax5.get_xticks()

# Skip every second tick
new_x2_ticks = x2_ticks[::1]

ax5.set_xticks(new_x2_ticks)



line_width = 1.5

ax4.plot(time_data, control_p1, linewidth=line_width, color=p1_color, label='Control P1', alpha = 0)
ax4.plot(time_data, control_p2, linewidth=line_width, color=p2_color, label='Control P2',alpha = 0)
ax4.plot(time_data, control_t1, linewidth=line_width, color=t1_color, label='Control T1',alpha = 0)
ax4.plot(time_data, control_t2, linewidth=line_width, color=t2_color, label='Control T2',alpha = 0)
ax4.set_ylabel(r'Control Signal (kV)',color='white')
ax4.set_xlabel(r'Time (s)',color='white')
ax4.grid(True)

ax3.plot(time_data, voltage_p1, linewidth=line_width, color=p1_color, label='Control P1',alpha = 0)
ax3.plot(time_data, voltage_p2, linewidth=line_width, color=p2_color, label='Control P2',alpha = 0)
ax3.plot(time_data, voltage_t1, linewidth=line_width, color=t1_color, label='Control T1',alpha = 0)
ax3.plot(time_data, voltage_t2, linewidth=line_width, color=t2_color, label='Control T2',alpha = 0)
ax3.set_ylabel(r'RMS Voltage (V)',color='white')
ax3.set_xlabel(r'Time (s)',color='white')
ax3.grid(True)

# Subplot 2: HSL positions vs Time
ax1.plot(time_data, ref_pos_t1, color=(1, 1,1), linestyle='--', linewidth=line_width, label='Ref Pos t1')
ax1.plot(time_data, est_pos_t1, color=est_color,linestyle="--", linewidth=line_width, label='Est Pos t1',alpha = 0)

ax1.plot(time_data, true_pos_t1, linewidth=line_width,color=p1_color, label='True Pos t1',alpha = 0)
ax1.plot(time_data, ref_pos_t2, color=(1, 1,1), linestyle='--', linewidth=line_width, label='Ref Pos t1')
ax1.plot(time_data, est_pos_t2, color=est_color,linestyle="--", linewidth=line_width, label='Est Pos t1',alpha = 0)

ax1.plot(time_data, true_pos_t2, linewidth=line_width,color=p2_color, label='True Pos t1',alpha = 0)
ax1.set_ylabel(r'HASEL Pos. (mm)',color='white')
ax1.set_xlabel(r'Time (s)',color='white')
ax1.grid(True)

ax2.plot(time_data, ref_pos_p1, color=(1, 1,1), linestyle='--', linewidth=line_width, label='Ref Pos P1')
ax2.plot(time_data, est_pos_p1, color=est_color,linestyle="--", linewidth=line_width, label='Est Pos P1',alpha = 0)

ax2.plot(time_data, true_pos_p1, linewidth=line_width,color=t1_color, label='True Pos P1',alpha = 0)
ax2.plot(time_data, ref_pos_p2, color=(1, 1,1), linestyle='--', linewidth=line_width, label='Ref Pos P1')
ax2.plot(time_data, est_pos_p2, color=est_color,linestyle="--", linewidth=line_width, label='Est Pos P1',alpha = 0)

ax2.plot(time_data, true_pos_p2, linewidth=line_width,color=t2_color, label='True Pos P1',alpha = 0)
ax2.set_ylabel(r'HASEL Pos. (mm)',color='white')
ax2.set_xlabel(r'Time (s)',color='white')
ax2.grid(True)

ax1.set_facecolor('black')
ax2.set_facecolor('black')
ax3.set_facecolor('black')
ax4.set_facecolor('black')
ax5.set_facecolor('black')
ax1.spines['bottom'].set_edgecolor('white')
ax1.spines['top'].set_edgecolor('white')
ax1.spines['right'].set_edgecolor('white')
ax1.spines['left'].set_edgecolor('white')
ax2.spines['bottom'].set_edgecolor('white')
ax2.spines['top'].set_edgecolor('white')
ax2.spines['right'].set_edgecolor('white')
ax2.spines['left'].set_edgecolor('white')
ax3.spines['bottom'].set_edgecolor('white')
ax3.spines['top'].set_edgecolor('white')
ax3.spines['right'].set_edgecolor('white')
ax3.spines['left'].set_edgecolor('white')
ax4.spines['bottom'].set_edgecolor('white')
ax4.spines['top'].set_edgecolor('white')
ax4.spines['right'].set_edgecolor('white')
ax4.spines['left'].set_edgecolor('white')
ax5.spines['bottom'].set_edgecolor('white')
ax5.spines['top'].set_edgecolor('white')
ax5.spines['right'].set_edgecolor('white')

legend_elements = [
    Line2D([0], [0], color=(1, 1,1), linestyle='--', lw=2, label='Reference'),
    Line2D([0], [0], color=true_color, lw=2, label='True'),   
    Line2D([0], [0], color=est_color, linestyle='--', lw=2, label='Estimated'),
    Line2D([0], [0], color=p1_color, lw=2, label='$\phi_1$ (True)'),
    Line2D([0], [0], color=p2_color, lw=2, label='$\phi_2$ (True)'),
    Line2D([0], [0], color=t1_color, lw=2, label='$\\theta_1$ (True)'),
    Line2D([0], [0], color=t2_color, lw=2, label='$\\theta_2$ (True)')
]

legend = fig.legend(handles=legend_elements, loc='lower left', handlelength=2,ncol=7, bbox_to_anchor=(-0.005, -0.01),edgecolor="white")
legend.get_frame().set_alpha(None)
legend.get_frame().set_facecolor((0, 0, 1, 0.1))
for text in legend.get_texts():
    text.set_color("white")
# Define a formatter to use for the tick labels
formatter = mticker.FormatStrFormatter('%.1f')
ax5.zaxis.set_major_formatter(formatter)
ax5.xaxis.set_tick_params(pad=0)  # Adjusting for x-axis
ax5.yaxis.set_tick_params(pad=-1.5)  # Adjusting for y-axis
ax5.zaxis.set_tick_params(pad=0)  # Adjusting for z-axis
ax5.xaxis.labelpad = -4  # Adjusting for x-axis label
ax5.yaxis.labelpad = -4 # Adjusting for y-axis label
ax5.zaxis.labelpad = -2  # Adjusting for z-axis label
# Apply the formatter to each of the axes
for ax in [ax1, ax2, ax3, ax4, ax5]:
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_formatter(formatter)

ax5.zaxis.set_major_formatter(formatter)

ax1.tick_params(colors='white')
ax2.tick_params(colors='white')
ax3.tick_params(colors='white')
ax4.tick_params(colors='white')
ax5.tick_params(colors='white')

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
fig.subplots_adjust(
    top=0.986,
    bottom=0.195,
    left=0.038,
    right=0.99,
    hspace=0.412,
    wspace=0.275
)
# Plotting configuration
line_width = 2

# Create empty line objects
l1, = ax1.plot([], [], color=(1, 1,1), linestyle='--', linewidth=line_width)
l2, = ax1.plot([], [], linewidth=line_width, color=p1_color)
l3, = ax1.plot([], [], color=(1, 1,1), linestyle='--', linewidth=line_width)
l4, = ax1.plot([], [], linewidth=line_width, color=p2_color)


# Note: 3D line plotting in animations can be trickier
l9, = ax5.plot([], [], [], color=est_color,linestyle = "--", linewidth=2)
l10, = ax5.plot([], [], [], color=true_color, linewidth=2)


l5, = ax2.plot([], [], color=(1, 1,1), linestyle='--', linewidth=line_width)
l6, = ax2.plot([], [], linewidth=line_width, color=t1_color)
l7, = ax2.plot([], [], color=(1, 1,1), linestyle='--', linewidth=line_width)
l8, = ax2.plot([], [], linewidth=line_width, color=t2_color)

l19, = ax1.plot([], [], color=est_color, linestyle="--", linewidth=line_width, label='Est Pos t1')
l20, = ax1.plot([], [], linewidth=line_width, color=p1_color, label='True Pos t1')
l21, = ax1.plot([], [], color=est_color, linestyle="--", linewidth=line_width, label='Est Pos t2')
l22, = ax1.plot([], [], linewidth=line_width, color=p2_color, label='True Pos t2')

# Create empty line objects for ax2
l23, = ax2.plot([], [], color=est_color, linestyle="--", linewidth=line_width, label='Est Pos P1')
l24, = ax2.plot([], [], linewidth=line_width, color=t1_color, label='True Pos P1')
l25, = ax2.plot([], [], color=est_color, linestyle="--", linewidth=line_width, label='Est Pos P2')
l26, = ax2.plot([], [], linewidth=line_width, color=t2_color, label='True Pos P2')


l11, = ax4.plot([], [], linewidth=line_width, color=p1_color, label='Control P1')
l12, = ax4.plot([], [], linewidth=line_width, color=p2_color, label='Control P2')
l13, = ax4.plot([], [], linewidth=line_width, color=t1_color, label='Control T1')
l14, = ax4.plot([], [], linewidth=line_width, color=t2_color, label='Control T2')

# Create empty line objects for ax3
l15, = ax3.plot([], [], linewidth=line_width, color=p1_color, label='Voltage P1')
l16, = ax3.plot([], [], linewidth=line_width, color=p2_color, label='Voltage P2')
l17, = ax3.plot([], [], linewidth=line_width, color=t1_color, label='Voltage T1')
l18, = ax3.plot([], [], linewidth=line_width, color=t2_color, label='Voltage T2')

progress_bar = tqdm(total=len(time_data), desc="Rendering", unit="frame")

def init():
    #l1, = ax1.plot([], [], 'k--', linewidth=line_width)
    l2, = ax1.plot([], [], linewidth=line_width, color=p1_color)
    #l3, = ax1.plot([], [], 'k--', linewidth=line_width)
    l4, = ax1.plot([], [], linewidth=line_width, color=p2_color)

    #l9, = ax5.plot([], [], [], 'k--', linewidth=2)
    l9, = ax5.plot([], [], [], color=est_color,linestyle="--", linewidth=2)

    l10, = ax5.plot([], [], [], color=true_color, linewidth=2)


    #l5, = ax2.plot([], [], 'k--', linewidth=line_width)
    l6, = ax2.plot([], [], linewidth=line_width, color=t1_color)
    #l7, = ax2.plot([], [], 'k--', linewidth=line_width)
    l8, = ax2.plot([], [], linewidth=line_width, color=t2_color)


    l11, = ax4.plot([], [], linewidth=line_width, color=p1_color, label='Control P1')
    l12, = ax4.plot([], [], linewidth=line_width, color=p2_color, label='Control P2')
    l13, = ax4.plot([], [], linewidth=line_width, color=t1_color, label='Control T1')
    l14, = ax4.plot([], [], linewidth=line_width, color=t2_color, label='Control T2')

    # Create empty line objects for ax3
    l15, = ax3.plot([], [], linewidth=line_width, color=p1_color, label='Voltage P1')
    l16, = ax3.plot([], [], linewidth=line_width, color=p2_color, label='Voltage P2')
    l17, = ax3.plot([], [], linewidth=line_width, color=t1_color, label='Voltage T1')
    l18, = ax3.plot([], [], linewidth=line_width, color=t2_color, label='Voltage T2')

    l19, = ax1.plot([], [], color=est_color, linestyle="--", linewidth=line_width, label='Est Pos t1')
    l20, = ax1.plot([], [], linewidth=line_width, color=p1_color, label='True Pos t1')
    l21, = ax1.plot([], [], color=est_color, linestyle="--", linewidth=line_width, label='Est Pos t2')
    l22, = ax1.plot([], [], linewidth=line_width, color=p2_color, label='True Pos t2')

    # Create empty line objects for ax2
    l23, = ax2.plot([], [], color=est_color, linestyle="--", linewidth=line_width, label='Est Pos P1')
    l24, = ax2.plot([], [], linewidth=line_width, color=t1_color, label='True Pos P1')
    l25, = ax2.plot([], [], color=est_color, linestyle="--", linewidth=line_width, label='Est Pos P2')
    l26, = ax2.plot([], [], linewidth=line_width, color=t2_color, label='True Pos P2')

    return l19, l20, l21, l22, l23, l24, l25, l26, l9,l10,l11, l12, l13, l14, l15, l16, l17, l18
    
   

def update(frame):
    #l1.set_data(time_data[:frame], ref_pos_t1[:frame])
    #l2.set_data(time_data[:frame], true_pos_t1[:frame])
    #l3.set_data(time_data[:frame], ref_pos_t2[:frame])
    #l4.set_data(time_data[:frame], true_pos_t2[:frame])
    
# For 3D data, we need to set both 2D data and z-data
    #l9.set_data(-x_ref[:frame], y_ref[:frame])
    #l9.set_3d_properties(z_ref[:frame])
    
    l9.set_data(x_est[:frame], y_est[:frame])
    l9.set_3d_properties(z_est[:frame])
    l10.set_data(x_true[:frame], y_true[:frame])
    l10.set_3d_properties(z_true[:frame])

    #l5.set_data(time_data[:frame], ref_pos_p1[:frame])
    #l6.set_data(time_data[:frame], true_pos_p1[:frame])
    #l7.set_data(time_data[:frame], ref_pos_p2[:frame])
    #l8.set_data(time_data[:frame], true_pos_p2[:frame])


    l11.set_data(time_data[:frame], control_p1[:frame])
    l12.set_data(time_data[:frame], control_p2[:frame])
    l13.set_data(time_data[:frame], control_t1[:frame])
    l14.set_data(time_data[:frame], control_t2[:frame])

    l15.set_data(time_data[:frame], voltage_p1[:frame])
    l16.set_data(time_data[:frame], voltage_p2[:frame])
    l17.set_data(time_data[:frame], voltage_t1[:frame])
    l18.set_data(time_data[:frame], voltage_t2[:frame])
    
    
    l19.set_data(time_data[:frame], est_pos_t1[:frame])
    l20.set_data(time_data[:frame], true_pos_t1[:frame])
    l21.set_data(time_data[:frame], est_pos_t2[:frame])
    l22.set_data(time_data[:frame], true_pos_t2[:frame])

    l23.set_data(time_data[:frame], est_pos_p1[:frame])
    l24.set_data(time_data[:frame], true_pos_p1[:frame])
    l25.set_data(time_data[:frame], est_pos_p2[:frame])
    l26.set_data(time_data[:frame], true_pos_p2[:frame])
    
    progress_bar.update(1)

    return l19, l20, l21, l22, l23, l24, l25, l26,l9, l10,l11, l12, l13, l14, l15, l16, l17, l18


ani = FuncAnimation(fig, update, frames=len(time_data)/19, init_func=init, blit=True)

ani.save('animated_plot_ss_lemni.mov', writer='ffmpeg', fps=30)

progress_bar.close()  # Close the progress bar when done

fig.savefig(save_location, dpi=1500)  # 1000 = 10 * 100
print(save_location)
plt.show()
