
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import math
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm
from matplotlib.animation import FFMpegWriter
import h5py

# Define the window size for calculating RMS values and moving average filter
rms_window_size = 200
moving_average_window_size = 1  # Adjust the window size for the moving average filter


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

# Plot ref, actual, and estimated angles
A = 6  # Want figures to be A6
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig, axs = plt.subplots(1, 1, figsize=(13, 9))  # 1 row, 2 columns
plt.rcParams['axes.labelsize'] = 8   # Set x and y labels fontsize
plt.rcParams['legend.fontsize'] = 8  # Set legend fontsize
plt.rcParams['xtick.labelsize'] = 6  # Set x tick labels fontsize
plt.rcParams['ytick.labelsize'] = 6  # Set y tick labels fontsize

line_width = 2.5

est_color='#FF6666'
true_color = '#00008b'
true_color = '#99FF99'
t2_color = '#80FF00'
t1_color = '#33FF99'
p2_color = '#33FFFF'
p1_color = '#66B2FF'

axs.plot(adjusted_time, adj_displacement, linewidth=line_width, color='r')
axs.plot(adjusted_time, est_displ_1, linewidth=line_width, color=p1_color)
axs.set_xlabel(r'Time (s)', weight='bold')  # X-axis label with increased font size and bold
axs.set_ylabel(r'Displacement (mm)')  # Y-axis label with increased font size and bold
axs.grid(True)  # Add grid with dashed lines




axs.tick_params(colors='white')

plt.style.use('dark_background')


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




axs.set_facecolor('black')

axs.spines['bottom'].set_edgecolor('white')
axs.spines['top'].set_edgecolor('white')
axs.spines['right'].set_edgecolor('white')
axs.spines['left'].set_edgecolor('white')


# Create empty line objects
l1, = axs.plot([], [], 'k--', linewidth=line_width, color='r')
l2, = axs.plot([], [], linewidth=line_width, color=p1_color)


progress_bar = tqdm(total=len(adjusted_time), desc="Rendering", unit="frame")

def init():
    # l1, = axs.plot([], [], 'k--', linewidth=line_width, color='r')
    # l2, = axs.plot([], [], linewidth=line_width, color=p1_color)
    # return l1, l2
    pass

    
   

def update(frame):
    l1.set_data(adjusted_time[:frame], adj_displacement[:frame])
    l2.set_data(adjusted_time[:frame], est_displ_1[:frame])

    progress_bar.update(1)
    return l1, l2


ani = FuncAnimation(fig, update, frames=len(adjusted_time/20), init_func=init, blit=True)

ani.save('chirp.mov', writer='ffmpeg', fps=30)

progress_bar.close()  # Close the progress bar when done

fig.savefig('chirp', dpi=1500)  # 1000 = 10 * 100
plt.show()
