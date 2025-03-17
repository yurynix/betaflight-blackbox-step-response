import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

# Read the CSV file
df = pd.read_csv('log.01.csv')

# Calculate actual log rate from time column
time_us = df[' time (us)'].values
log_rate = 1e6 / np.median(np.diff(time_us))  # Calculate rate in Hz
print(f"Log rate: {log_rate:.1f} Hz")

def calculate_step_response(setpoint, gyro, log_rate, smooth_factor=2, axis_name="", y_correction=True):
    print(f"\nProcessing {axis_name}...")
    
    # Basic signal stats
    print(f"Setpoint range: {np.min(setpoint):.1f} to {np.max(setpoint):.1f}")
    print(f"Gyro range: {np.min(gyro):.1f} to {np.max(gyro):.1f}")
    
    # Calculate smoothing windows based on log rate
    base_windows = [1, 20, 40, 60]
    window_ms = base_windows[smooth_factor] * (log_rate / 1000)  # Scale window with log rate
    window = int(window_ms)
    window = window + (1 - window % 2)  # Ensure odd window size for savgol
    
    # Apply smoothing with edge handling
    gyro_smooth = savgol_filter(gyro, window, 3)
    
    # Parameters for segmentation
    min_input = 15  # Reduced minimum step size
    segment_length = int(log_rate * 2)  # 2 second segments
    window_size = int(0.5 * log_rate)  # 500ms window for step response
    t = np.linspace(0, 500, window_size)  # Time vector in milliseconds
    
    # Calculate subsample factor based on file duration
    file_dur_sec = len(setpoint) / log_rate
    if file_dur_sec <= 20:
        subsample_factor = 10
    elif file_dur_sec <= 60:
        subsample_factor = 7
    else:
        subsample_factor = 3
        
    # Create segments
    n_segments = len(setpoint) // segment_length
    valid_responses = []
    
    print(f"Processing {n_segments} segments...")
    
    for i in range(n_segments):
        start_idx = i * segment_length
        end_idx = start_idx + segment_length
        
        sp_seg = setpoint[start_idx:end_idx]
        gy_seg = gyro_smooth[start_idx:end_idx]
        
        seg_max = np.max(sp_seg)
        if seg_max < min_input:
            continue
            
        print(f"Segment {i}: max input = {seg_max:.1f}")
            
        # Apply Hann window
        window_func = np.hanning(len(sp_seg))
        sp_windowed = sp_seg * window_func
        gy_windowed = gy_seg * window_func
        
        # Pad signals
        pad_length = 100
        sp_padded = np.pad(sp_windowed, (0, pad_length))
        gy_padded = np.pad(gy_windowed, (0, pad_length))
        
        # Calculate frequency response
        H = np.fft.fft(sp_padded) / len(sp_padded)
        G = np.fft.fft(gy_padded) / len(gy_padded)
        
        # Calculate impulse response
        imp = np.real(np.fft.ifft(G * np.conj(H) / (H * np.conj(H) + 0.001))) * pad_length  # Increased regularization
        step = np.cumsum(imp)
        
        # Trim to window size
        step = step[:window_size]
        
        # Normalize step to [0,1] range
        step = (step - np.min(step)) / (np.max(step) - np.min(step))
        
        # Validate steady state
        steady_state_mask = (t > 200) & (t < 500)
        steady_state = step[steady_state_mask]
        steady_state_mean = np.mean(steady_state)
        
        print(f"  Steady state mean: {steady_state_mean:.3f}")
        
        # Apply Y correction if needed
        if y_correction and abs(steady_state_mean - 1) > 0.1:  # Only correct if more than 10% off
            y_offset = 1 - steady_state_mean
            step = step * (1 + y_offset)
            steady_state = step[steady_state_mask]
            steady_state_mean = np.mean(steady_state)
            print(f"  After correction: {steady_state_mean:.3f}")
        
        # Validate step response with more lenient criteria
        if 0.3 < steady_state_mean < 2 and np.min(steady_state) > 0.2 and np.max(steady_state) < 2.5:
            valid_responses.append(step)
            print(f"  Valid response found!")
    
    print(f"Found {len(valid_responses)} valid step responses")
    
    if not valid_responses:
        return None, None, None, None, None
        
    # Average all valid responses
    avg_response = np.mean(valid_responses, axis=0)
    
    # Calculate steady state value using last 60% of response
    steady_state_value = np.mean(avg_response[int(0.4 * len(avg_response)):])
    
    # Calculate rise time (time to reach 90% of steady state)
    rise_threshold = 0.9 * steady_state_value
    rise_time_idx = np.where(avg_response >= rise_threshold)[0][0]
    rise_time = t[rise_time_idx]
    
    # Calculate overshoot
    max_value = np.max(avg_response)
    overshoot = max(0, (max_value - steady_state_value) / steady_state_value * 100)
    
    # Calculate settling time (within 2% of steady state)
    settling_band = 0.02 * steady_state_value
    settling_mask = np.abs(avg_response - steady_state_value) <= settling_band
    # Find the last time we exit the settling band, then the first time we enter it permanently
    settling_crossings = np.where(np.diff(settling_mask.astype(int)) != 0)[0]
    if len(settling_crossings) > 0:
        settling_time = t[settling_crossings[-1] + 1]
    else:
        settling_time = t[0]  # No settling time if we're always within the band
    
    metrics = {
        'rise_time': rise_time,
        'overshoot': overshoot,
        'settling_time': settling_time
    }
    
    return t, metrics, avg_response

# Create figure with three subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
plt.subplots_adjust(hspace=0.3)

# Common axis settings
def setup_axis(ax, title, metrics=None):
    ax.grid(True, which='both', alpha=0.2)
    ax.set_ylabel(title)
    ax.set_ylim(-0.2, 1.5)
    ax.set_xlim(0, 500)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    if metrics is not None:
        # Rise time marker
        rise_time = metrics['rise_time']
        ax.axvline(x=rise_time, color='green', linestyle='--', alpha=0.7)
        ax.text(rise_time + 5, 1.3, f'Rise: {rise_time:.1f}ms', rotation=0, color='green')
        
        # Settling time marker
        settling_time = metrics['settling_time']
        ax.axvline(x=settling_time, color='blue', linestyle='--', alpha=0.7)
        ax.text(settling_time + 5, 1.1, f'Settling: {settling_time:.1f}ms', rotation=0, color='blue')
        
        # Overshoot text - centered
        overshoot = metrics['overshoot']
        ax.text(250, 1.3, f'Overshoot: {overshoot:.1f}%', color='red', 
                horizontalalignment='center', verticalalignment='center')

# Plot roll step response
t, metrics, resp = calculate_step_response(df[' rcCommand[0]'].values, df[' gyroADC[0]'].values, log_rate, axis_name="Roll")
if t is not None:
    ax1.plot(t, resp, color='red', alpha=0.7)
    setup_axis(ax1, 'Roll', metrics)

# Plot pitch step response
t, metrics, resp = calculate_step_response(df[' rcCommand[1]'].values, df[' gyroADC[1]'].values, log_rate, axis_name="Pitch")
if t is not None:
    ax2.plot(t, resp, color='red', alpha=0.7)
    setup_axis(ax2, 'Pitch', metrics)

# Plot yaw step response
t, metrics, resp = calculate_step_response(df[' rcCommand[2]'].values, df[' gyroADC[2]'].values, log_rate, axis_name="Yaw")
if t is not None:
    ax3.plot(t, resp, color='red', alpha=0.7)
    setup_axis(ax3, 'Yaw', metrics)
    ax3.set_xlabel('Time (ms)')

plt.savefig('step_response.png', dpi=300, bbox_inches='tight') 