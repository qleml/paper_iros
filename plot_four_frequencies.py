import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from matplotlib.lines import Line2D

# Define the window size for calculating RMS values and moving average filter
rms_window_size = 200
moving_average_window_size = 10  # Adjust the window size for the moving average filter

# 10_Sine_34gg_2kHz_1M-1M_26-July-2023_12-18-37 für Fig 9 und 10
# 10_Sine_34gg_2kHz_TestPCB_26-July-2023_16-52-48 für Fig. 12 (8 Hz)
# 10_Sine_34gg_2kHz_UpsideDown_26-July-2023_21-10-06 für Fig. 11 (1 Hz)
# 10_Sine_34gg_2kHz_TestPCB_26-July-2023_16-49-14 für Fig. 14 (20 Hz)
# 10_Sine_34gg_2kHz_TestPCB_26-July-2023_16-51-12 für Fig. 13 (10 Hz) Matlab benutzen

file_paths = [
    'MLogs/10_Sine_34gg_2kHz_UpsideDown_26-July-2023_21-10-06/data.mat',
    'MLogs/10_Sine_34gg_2kHz_TestPCB_26-July-2023_16-52-48/data.mat',
    'MLogs/10_Sine_34gg_2kHz_TestPCB_26-July-2023_16-51-12/data.mat',
    'MLogs/10_Sine_34gg_2kHz_TestPCB_26-July-2023_16-49-14/data.mat'
]

A = 6 
plt.rc('text', usetex=True)
plt.rc('font', family='serif', weight='bold')

t2_color = 'y'
t1_color = '#2ca02c'
p2_color = 'c'
p1_color = '#1f77b4'

plt.rcParams['axes.labelsize'] = 25   # Set x and y labels fontsize
plt.rcParams['legend.fontsize'] = 16  # Set legend fontsize
plt.rcParams['xtick.labelsize'] = 20  # Set x tick labels fontsize
plt.rcParams['ytick.labelsize'] = 20  # Set y tick labels fontsize
plt.rcParams['grid.linewidth'] = 1.5
plt.rcParams['axes.linewidth'] = 1.5

line_width = 2.5

fig, axs = plt.subplots(2, 2, figsize=(16, 9))  # 1 row, 2 columns

for j, file_path in enumerate(file_paths):
        
    with h5py.File(file_path, 'r') as file:
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
    rms_voltage_values = pd.Series(rms_voltage_values, dtype=float)

    # if j==3:
    #     adj_displacement = adj_displacement[adjusted_time >= 0.1]
    #     rms_voltage_values = rms_voltage_values[adjusted_time >= 0.1]
    #     adjusted_time = adjusted_time[adjusted_time >= 0.1]
    #     adjusted_time -= 0.1

    if rms_voltage_values.isna().any():
            print(f"NaN values found in window ALDER")
    rms_voltage_values = rms_voltage_values.rolling(window=moving_average_window_size, min_periods=1).mean()
    if rms_voltage_values.isna().any():
            print(f"NaN values found in window ")


    coefficients = np.polyfit(rms_voltage_values, adj_displacement, degree)
    
    # coefficients_2 = [-125.6595, 1734.503, -7973.444, 12207.37]
    # coefficients_2 = [-14.53913, 182.8583, -766.8713, 1072.563]
    # coefficients_2 = [7.427259, -89.15698, 355.8401, -472.8132]

    if j==2:
        coefficients = [7.597979, -91.58709, 367.074, -489.5696] # 10 Hz


    est_displ = np.polyval(coefficients, rms_voltage_values)

    offset = np.min(adj_displacement)
    adj_displacement -= offset
    est_displ -= offset

    if j==0:
         adj_displacement *= 2.25
         est_displ *= 2.25

    axs[j // 2][j % 2].plot(adjusted_time, adj_displacement, linewidth=line_width, color='r')
    axs[j // 2][j % 2].plot(adjusted_time, est_displ, linewidth=line_width, color=p1_color)

    if j==2 or j==3:
        axs[j // 2][j % 2].set_xlabel(r'Time (s)', weight='bold')  # X-axis label with increased font size and bold
    if j==0 or j==2:
        axs[j // 2][j % 2].set_ylabel(r'Displacement (mm)')  # Y-axis label with increased font size and bold
    axs[j // 2][j % 2].grid(True)  # Add grid with dashed lines
    print("yey")


legend_elements = [
    Line2D([0], [0], color=p1_color, lw=2, label='Estimated Displacement (Method 1)'),
    Line2D([0], [0], color='r', lw=2, label='Ground Truth Displacement'),
]

fig.legend(handles=legend_elements, loc='upper center', handlelength=2,ncol=7, bbox_to_anchor=(0.5, 1.01), fontsize=18)

fig.subplots_adjust(
    top=0.925,
    bottom=0.315,
    left=0.075,
    right=0.98,
    hspace=0.2,
    wspace=0.105
)

plt.savefig('filtered-four.pdf')
# plt.legend(['Actual Displacement', 'Estimated Displacement'])
# plt.title('Estimated Displacement', fontsize=25)


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