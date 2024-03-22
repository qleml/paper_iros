import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm
from matplotlib.animation import FFMpegWriter

# Define the window size for calculating RMS values and moving average filter
rms_window_size = 200
moving_average_window_size = 1  # Adjust the window size for the moving average filter

# 10_Sine_34gg_2kHz_1M-1M_26-July-2023_12-18-37 für Fig 9 und 10

# 10_Vogt_34gg_2kHz_TestPCB_26-July-2023_15-36-24 Fig. 15
# 10_LFT_34gg_2kHz_UpsideDown_26-July-2023_21-17-32 Fig. 16

with h5py.File('MLogs/10_Vogt_34gg_2kHz_TestPCB_26-July-2023_15-36-24/data.mat', 'r') as file:
#with h5py.File('MLogs/10_LFT_34gg_2kHz_UpsideDown_26-July-2023_21-17-32/data.mat', 'r') as file:
    # Extract the data
    data = {}
    for key, value in file.items():
        # Check if the value is 1-dimensional
        if len(value.shape) == 1:
            data[key] = value[:]
        else:
            # If not 1-dimensional, convert to a list of 1-dimensional arrays
            for i in range(value.shape[1]):
                data[f"{key}_{i+1}"] = value[:, i]

# Convert to DataFrame
df = pd.DataFrame(data)

# Extract the columns from the data
time = df.iloc[0, :]
cmd_voltage = df.iloc[1, :]
displacement = df.iloc[2, :]
ss_voltage = df.iloc[3, :]

# Factor for laser displacement and reference
displacement = displacement * 4 + 30
displacement = displacement - np.mean(displacement.iloc[:50])

# Calculate the number of samples and the number of RMS calculations
num_samples = len(time)
num_rms = num_samples // rms_window_size

# Initialize arrays for RMS values and adjusted time and error
adjusted_time = np.zeros(num_rms)
adj_displacement = np.zeros(num_rms)
rms_voltage_values = np.zeros(num_rms)

#impedance = np.zeros(num_rms)

displacement = displacement.rolling(window=moving_average_window_size, min_periods=1).mean()

# Calculate RMS values and adjusted time
for i in range(num_rms):
    start_index = i * rms_window_size
    end_index = (i + 1) * rms_window_size - 1   

    # Calculate RMS values
    rms_voltage_values[i] = np.sqrt(np.mean(ss_voltage.iloc[start_index:end_index] ** 2))

    # Adjusted time is set to the middle time value of the window
    adjusted_time[i] = np.mean(time.iloc[start_index:end_index])
    adj_displacement[i] = np.mean(displacement.iloc[start_index:end_index])

degree = 3
rms_voltage_values = pd.Series(rms_voltage_values)

if rms_voltage_values.isna().any():
        print(f"NaN values found in window ALDER")

rms_voltage_values = rms_voltage_values.rolling(window=moving_average_window_size, min_periods=1).mean()

if rms_voltage_values.isna().any():
        print(f"NaN values found in window ")


coefficients_1 = np.polyfit(rms_voltage_values, adj_displacement, degree)
print(coefficients_1)
### if VOGT File change coeffs:
#coefficients_1 = [63.78575, -236.9001, 278.6847, -101.7]
###
est_displ_1 = np.polyval(coefficients_1, rms_voltage_values)

offset = np.min(adj_displacement)
adj_displacement -= offset
est_displ_1 -= offset
##################################################
# adj_displacement = adj_displacement[adjusted_time >= 5]
# est_displ_1 = est_displ_1[adjusted_time >= 5]
# est_displ_2 = est_displ_2[adjusted_time >= 5]
# adjusted_time = adjusted_time[adjusted_time >= 5]
# adjusted_time -= 5

#start = 7
#end = 13

# ofset the x-axis by start time
#adjusted_time -= start

A = 6 
plt.rc('text', usetex=True)
plt.rc('font', family='serif', weight='bold')

t2_color = 'y'
t1_color = '#2ca02c'
p2_color = 'c'
p1_color = '#1f77b4'

plt.rcParams['axes.labelsize'] = 25   # Set x and y labels fontsize
plt.rcParams['legend.fontsize'] = 8  # Set legend fontsize
plt.rcParams['xtick.labelsize'] = 20  # Set x tick labels fontsize
plt.rcParams['ytick.labelsize'] = 20  # Set y tick labels fontsize
plt.rcParams['grid.linewidth'] = 1.5
plt.rcParams['axes.linewidth'] = 1.5

line_width = 2.5

fig, axs = plt.subplots(1, 1, figsize=(13, 9))  # 1 row, 2 columns

axs.plot(adjusted_time, adj_displacement, linewidth=line_width, color='r')
axs.plot(adjusted_time, est_displ_1, linewidth=line_width, color=p1_color)
axs.set_xlabel(r'Time (s)', weight='bold')  # X-axis label with increased font size and bold
axs.set_ylabel(r'Displacement (mm)')  # Y-axis label with increased font size and bold
axs.grid(True)  # Add grid with dashed lines


plt.xlabel(r'Time (s)', weight='bold')  # X-axis label with increased font size and bold
plt.grid(True)  # Add grid with dashed lines


legend_elements = [
    Line2D([0], [0], color=p1_color, lw=2, label='Estim. Displacement (Method 1)'),
    Line2D([0], [0], color='r', lw=2, label='Ground Truth Displacement'),
]

fig.legend(handles=legend_elements, loc='upper center', handlelength=2,ncol=7, bbox_to_anchor=(0.5, 1.01), fontsize=18)
fig.subplots_adjust(
    top=0.925,
    bottom=0.615,
    left=0.3,
    right=0.98,
    hspace=0.2,
    wspace=0.105
)


l1, = axs.plot([], [], color=(1, 1,1), linestyle='--', linewidth=line_width)
l2, = axs.plot([], [], linewidth=line_width, color=p1_color)

def init():
    l2, = axs.plot([], [], linewidth=line_width, color=p1_color)
    return l2

def update(frame):
    l1 = axs.plot(adjusted_time, adj_displacement, linewidth=line_width, color='r')
    l2 = axs.plot(adjusted_time, est_displ_1, linewidth=line_width, color=p1_color)  


plt.savefig('vogt-filtered-movavg1.pdf')
# plt.legend(['Actual Displacement', 'Estimated Displacement'])
# plt.title('Estimated Displacement', fontsize=25)

ani = FuncAnimation(fig, update, frames=len(adjusted_time)/10, init_func=init, blit=True)

ani.save('animated_plot_ss_lemni.mov', writer='ffmpeg', fps=30)
#################################################### PLOT POLYNOMIAL FIT 

# residuals = est_displ - adj_displacement

# # Calculate MSE and RMSE
# mse = np.mean(residuals ** 2)
# rmse = np.sqrt(mse)

# # Create a new figure
# fig2 = plt.figure(figsize=(15, 8))

# # Plot data
# plt.plot(impedance, adj_displacement, 'bx', label='Measured Data Points')
# plt.plot(impedance, est_displ, 'r-', linewidth=3, label='Polynomial Fit: Coefficients: ')

# # Add labels and title
# plt.xlabel('RMS Voltage [V]')
# plt.ylabel('Displacement [mm]')
# plt.title('Mapping: RMSE {:.2f}'.format(rmse))
# plt.legend()
######################################################

# Show the plot
plt.show()