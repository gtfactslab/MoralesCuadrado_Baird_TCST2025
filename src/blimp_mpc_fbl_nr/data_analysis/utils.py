import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def calculate_overall_rmse(df):
    """
    Calculate the overall RMSE across x, y, z, and yaw compared to their reference values.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing the actual values and reference values.
    
    Returns:
    float: The overall RMSE across all dimensions.
    """
    
    # Extract actual and reference values as numpy arrays
    actual_values = df[['x', 'y', 'z', 'psi']].to_numpy()
    reference_values = df[['x_ref', 'y_ref', 'z_ref', 'psi_ref']].to_numpy()
    
    # Compute the squared differences
    squared_errors = (actual_values - reference_values) ** 2
    
    # Compute the mean of the sum of squared differences across all dimensions
    mse = np.mean(np.sum(squared_errors, axis=1))
    
    # Return the square root of the mean squared error (overall RMSE)
    overall_rmse = np.sqrt(mse)
    
    return overall_rmse


def plot_adjusted_avg_and_max_xyz_vs_reference(df):
    fig, axs = plt.subplots(3, 4, figsize=(20, 12), sharex=False)
    time_max = df['time'].max()
    time_min = df['time'].min()
    x_lim = (time_min-1, time_max+1)
    
    # Maximum values
    max_fx = 0.035
    max_fy = 0.035
    max_fz = 0.2
    max_tz = 0.001

    # Calculate average values
    avg_fx = df['fx'].mean()
    avg_fy = df['fy'].mean()
    avg_fz = df['fz'].mean()
    avg_tz = df['tauz'].mean()

    # Add some margin to cover both average and max values
    margin_fx = max(max_fx, abs(avg_fx)) * 1.2
    margin_fy = max(max_fy, abs(avg_fy)) * 1.2
    margin_fz = max(max_fz, abs(avg_fz)) * 1.2
    margin_tz = max(max_tz, abs(avg_tz)) * 1.2

    # Row 1: Plot x, y, z, psi vs references
    # plot x vs x_ref
    axs[0, 0].plot(df['time'], df['x'], label='x', color='red')
    axs[0, 0].plot(df['time'], df['x_ref'], label='x_ref', color='blue', linestyle='--')
    axs[0, 0].set_ylabel('x / x_ref')
    axs[0, 0].set_xlabel('time')
    axs[0, 0].legend()
    axs[0, 0].set_xlim(x_lim)

    # plot y vs y_ref
    axs[0, 1].plot(df['time'], df['y'], label='y', color='red')
    axs[0, 1].plot(df['time'], df['y_ref'], label='y_ref', color='blue', linestyle='--')
    axs[0, 1].set_ylabel('y / y_ref')
    axs[0, 1].set_xlabel('time')
    axs[0, 1].legend()
    axs[0, 1].set_xlim(x_lim)

    # plot z vs z_ref
    axs[0, 2].plot(df['time'], -1*df['z'], label='z', color='red')
    axs[0, 2].plot(df['time'], -1*df['z_ref'], label='z_ref', color='blue', linestyle='--')
    axs[0, 2].set_ylabel('z / z_ref')
    axs[0, 2].set_xlabel('time')
    axs[0, 2].legend()
    axs[0, 2].set_xlim(x_lim)
    axs[0, 2].set_ylim(0,-1*df['z'].min()+.1)


    # plot psi vs psi_ref
    axs[0, 3].plot(df['time'], df['psi'], label='psi', color='red')
    axs[0, 3].plot(df['time'], df['psi_ref'], label='psi_ref', color='blue', linestyle='--')
    axs[0, 3].set_ylabel('psi / psi_ref')
    axs[0, 3].set_xlabel('time')
    axs[0, 3].legend()
    axs[0, 3].set_xlim(x_lim)

    # Row 2: Plot cross comparisons (x vs y, x vs z, y vs z, time vs solve_time)
    # plot x vs y and x_ref vs y_ref
    axs[1, 0].plot(df['x'], df['y'], label='x vs y', color='red')
    axs[1, 0].plot(df['x_ref'], df['y_ref'], label='x_ref vs y_ref', color='blue', linestyle='--')
    axs[1, 0].set_ylabel('y')
    axs[1, 0].set_xlabel('x')
    axs[1, 0].legend()

    # plot x vs z and x_ref vs z_ref
    axs[1, 1].plot(df['x'], -1*df['z'], label='x vs z', color='red')
    axs[1, 1].plot(df['x_ref'], -1*df['z_ref'], label='x_ref vs z_ref', color='blue', linestyle='--')
    axs[1, 1].set_ylabel('z')
    axs[1, 1].set_xlabel('x')
    axs[1, 1].set_ylim(0,-1*df['z'].min()+.1)

    axs[1, 1].legend()

    # plot y vs z and y_ref vs z_ref
    axs[1, 2].plot(df['y'], -1*df['z'], label='y vs z', color='red')
    axs[1, 2].plot(df['y_ref'], -1*df['z_ref'], label='y_ref vs z_ref', color='blue', linestyle='--')
    axs[1, 2].set_ylabel('z')
    axs[1, 2].set_xlabel('y')
    axs[1, 2].set_ylim(0,-1*df['z'].min()+.1)

    axs[1, 2].legend()

    # plot time vs solve_time
    axs[1, 3].plot(df['time'], df['solve_time']* 1e-9, label='solve_time', color='purple')
    axs[1, 3].set_ylabel('solve_time')
    axs[1, 3].set_xlabel('time')
    axs[1, 3].legend()
    axs[1, 3].set_xlim(x_lim)

    # Row 3: Plot fx, fy, fz, tauz vs time
    # plot fx vs time
    axs[2, 0].plot(df['time'], df['fx'], label='fx', color='blue')
    axs[2, 0].axhline(y=max_fx, color='red', linestyle='--', label=f'+{max_fx}')
    axs[2, 0].axhline(y=-max_fx, color='red', linestyle='--', label=f'-{max_fx}')
    axs[2, 0].set_ylabel('fx')
    axs[2, 0].set_xlabel('time')
    axs[2, 0].legend()
    axs[2, 0].set_ylim((-margin_fx, margin_fx))
    axs[2, 0].set_xlim(x_lim)

    # plot fy vs time
    axs[2, 1].plot(df['time'], df['fy'], label='fy', color='orange')
    axs[2, 1].axhline(y=max_fy, color='red', linestyle='--', label=f'+{max_fy}')
    axs[2, 1].axhline(y=-max_fy, color='red', linestyle='--', label=f'-{max_fy}')
    axs[2, 1].set_ylabel('fy')
    axs[2, 1].set_xlabel('time')
    axs[2, 1].legend()
    axs[2, 1].set_ylim((-margin_fy, margin_fy))
    axs[2, 1].set_xlim(x_lim)

    # plot fz vs time
    axs[2, 2].plot(df['time'], df['fz'], label='fz', color='green')
    axs[2, 2].axhline(y=max_fz, color='red', linestyle='--', label=f'+{max_fz}')
    axs[2, 2].axhline(y=-max_fz, color='red', linestyle='--', label=f'-{max_fz}')
    axs[2, 2].set_ylabel('fz')
    axs[2, 2].set_xlabel('time')
    axs[2, 2].legend()
    axs[2, 2].set_ylim((-margin_fz, margin_fz))
    axs[2, 2].set_xlim(x_lim)

    # plot tauz vs time
    axs[2, 3].plot(df['time'], df['tauz'], label='tauz', color='purple')
    axs[2, 3].axhline(y=max_tz, color='red', linestyle='--', label=f'+{max_tz}')
    axs[2, 3].axhline(y=-max_tz, color='red', linestyle='--', label=f'-{max_tz}')
    axs[2, 3].set_ylabel('tauz')
    axs[2, 3].set_xlabel('time')
    axs[2, 3].legend()
    axs[2, 3].set_ylim((-margin_tz, margin_tz))
    axs[2, 3].set_xlim(x_lim)

    