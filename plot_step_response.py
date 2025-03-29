import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import os
import subprocess
import glob
from pathlib import Path
import tempfile
import shutil

def get_log_count(bbl_file):
    """Get the number of logs in a BBL file."""
    try:
        result = subprocess.run(['./blackbox_decode', '--stdout', '--debug', bbl_file],
                              capture_output=True, text=True)
        if result.returncode == 0:
            # Look for lines like "Index  Start offset  Size (bytes)"
            lines = result.stdout.split('\n')
            for i, line in enumerate(lines):
                if 'Index  Start offset  Size' in line:
                    # Count the number of non-empty lines that follow
                    count = 0
                    for j in range(i + 1, len(lines)):
                        if lines[j].strip():
                            count += 1
                        else:
                            break
                    return count
        return 1
    except Exception as e:
        print(f"Error getting log count: {e}")
        return 1

def convert_bbl_to_csv(bbl_file, log_index, temp_dir):
    """Convert BBL file to CSV using blackbox_decode with specific log index and return PID coefficients"""
    base_name = os.path.splitext(os.path.basename(bbl_file))[0]
    csv_file = os.path.join(temp_dir, f'{base_name}.{log_index:02d}.csv')
    headers_file = os.path.join(temp_dir, f'{base_name}.{log_index:02d}.headers.csv')
    
    try:
        # First run to get header information with --save-headers
        header_result = subprocess.run([
            './blackbox_decode', 
            '--save-headers',
            '--index', str(log_index),
            bbl_file
        ], capture_output=True, text=True)
        
        # Move headers file from logs dir to temp dir if it exists
        orig_headers_file = f'logs/{base_name}.{log_index:02d}.headers.csv'
        if os.path.exists(orig_headers_file):
            shutil.move(orig_headers_file, headers_file)
        
        # Initialize PID coefficients and settings
        pid_coeffs = {
            'roll': {'P': 0, 'I': 0, 'D': 0},
            'pitch': {'P': 0, 'I': 0, 'D': 0},
            'yaw': {'P': 0, 'I': 0, 'D': 0}
        }
        
        # Initialize settings dictionary
        settings = {
            'ff_weight': '0,0,0',
            'feedforward_boost': '0',
            'feedforward_smooth_factor': '0',
            'feedforward_jitter_factor': '0',
            'feedforward_max_rate_limit': '0',
            'dterm_lpf1_static_hz': '0',
            'dterm_lpf2_static_hz': '0',
            'gyro_lpf2_static_hz': '0',
            'dyn_notch_min_hz': '0',
            'dyn_notch_max_hz': '0',
            'iterm_windup': '0',
            'iterm_relax': '0',
            'iterm_relax_cutoff': '0',
            'd_min': '0,0,0',
            'd_max_gain': '0',
            'anti_gravity_gain': '0',
            'rc_smoothing_feedforward_cutoff': '0',
            'pidsum_limit': '0',
        }
        
        # Read PID values and settings from headers file if it exists
        if os.path.exists(headers_file):
            try:
                with open(headers_file, 'r') as f:
                    # Skip the header line
                    next(f)
                    for line in f:
                        # Split by comma and strip quotes
                        name, value = [x.strip().strip('"') for x in line.split(',', 1)]
                        
                        # Store PID values
                        if name == 'rollPID':
                            p, i, d = map(float, value.split(','))
                            pid_coeffs['roll'] = {'P': p, 'I': i, 'D': d}
                        elif name == 'pitchPID':
                            p, i, d = map(float, value.split(','))
                            pid_coeffs['pitch'] = {'P': p, 'I': i, 'D': d}
                        elif name == 'yawPID':
                            p, i, d = map(float, value.split(','))
                            pid_coeffs['yaw'] = {'P': p, 'I': i, 'D': d}
                        
                        # Store other relevant settings
                        elif name in settings:
                            settings[name] = value
                
                print(f"Found PID values: {pid_coeffs}")
            except Exception as e:
                print(f"Error reading headers file: {e}")
        
        # Now convert the file to CSV using stdout
        result = subprocess.run([
            './blackbox_decode', 
            '--index', str(log_index),
            '--stdout',
            bbl_file
        ], stdout=open(csv_file, 'w'), 
           stderr=subprocess.PIPE,
           text=True)
        
        if result.returncode == 0 and os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
            return csv_file, pid_coeffs, settings
        else:
            print(f"Error converting log {log_index}: {result.stderr}")
            return None, None, None
    except Exception as e:
        print(f"Error running blackbox_decode: {e}")
        return None, None, None

def process_log_file(csv_file, pid_coeffs=None, settings=None):
    """Process a single log file and return step responses"""
    print(f"\nProcessing file: {csv_file}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Print all columns
        print("\nAll columns:")
        for col in df.columns:
            print(f"  {col}")
        
        # Print first 5 rows of debug columns for debugging
        print("\nFirst 5 rows of debug columns:")
        debug_columns = [col for col in df.columns if 'debug' in col]
        print(df[debug_columns].head())
        print("\n")
        
        # Calculate actual log rate from time column
        time_us = df[' time (us)'].values
        log_rate = 1e6 / np.median(np.diff(time_us))
        print(f"Log rate: {log_rate:.1f} Hz")
        
        # Process each axis
        responses = {}
        axis_map = {0: 'roll', 1: 'pitch', 2: 'yaw'}
        
        for axis, (cmd_col, gyro_col) in enumerate([
            (' rcCommand[0]', ' gyroADC[0]'),
            (' rcCommand[1]', ' gyroADC[1]'),
            (' rcCommand[2]', ' gyroADC[2]')
        ]):
            if cmd_col in df.columns and gyro_col in df.columns:
                t, metrics, resp = calculate_step_response(
                    df[cmd_col].values,
                    df[gyro_col].values,
                    log_rate,
                    axis_name=f"Axis {axis}"
                )
                if t is not None:
                    # Get PID values for this axis
                    axis_name = axis_map[axis]
                    p = pid_coeffs[axis_name]['P'] if pid_coeffs else 0
                    i = pid_coeffs[axis_name]['I'] if pid_coeffs else 0
                    d = pid_coeffs[axis_name]['D'] if pid_coeffs else 0
                    responses[axis] = (t, metrics, resp, (p, i, d))
        
        return responses, settings
    except Exception as e:
        print(f"Error processing CSV file: {e}")
        return None, None

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
        return None, None, None
        
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

def setup_axis(ax, title):
    ax.grid(True, which='both', alpha=0.2)
    ax.set_ylabel(title)
    ax.set_ylim(-0.2, 1.5)
    ax.set_xlim(0, 500)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

def plot_responses(responses, output_file):
    """Plot step responses for each axis."""
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    # Leave less space between the warning and the subplots
    plt.subplots_adjust(hspace=0.3, top=0.85)

    # Settings dictionaries for each log
    all_settings = []
    
    # Color cycle for different logs
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    color_idx = 0
    
    # Create a temporary directory for the files
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Created temporary directory: {temp_dir}")
        
        # Process all BBL files in logs directory
        bbl_files = glob.glob('logs/*.bbl')
        for bbl_file in bbl_files:
            print(f"\nProcessing {bbl_file}...")
            log_count = get_log_count(bbl_file)
            print(f"Found {log_count} logs in {bbl_file}")
            
            # Process each log in the file
            for log_index in range(1, log_count + 1):
                print(f"\nProcessing log {log_index} of {log_count}...")
                try:
                    csv_file, pid_coeffs, settings = convert_bbl_to_csv(bbl_file, log_index, temp_dir)
                    
                    if csv_file and os.path.exists(csv_file):
                        try:
                            # Process the CSV file
                            file_responses, file_settings = process_log_file(csv_file, pid_coeffs, settings)
                            if file_responses:
                                # Store settings for warning text
                                if file_settings:
                                    all_settings.append(file_settings)
                                    
                                # Plot responses for each axis
                                axes = [ax1, ax2, ax3]
                                axis_names = ['Roll', 'Pitch', 'Yaw']
                                color = colors[color_idx % len(colors)]
                                base_name = os.path.basename(bbl_file)
                                
                                for axis, ax in enumerate(axes):
                                    if axis in file_responses:
                                        t, metrics, resp, pid_values = file_responses[axis]
                                        p, i, d = pid_values
                                        label = (f'{base_name} - '
                                               f'P={p:.1f} I={i:.1f} D={d:.1f}\n'
                                               f'Rise: {metrics["rise_time"]:.1f}ms, '
                                               f'Settling: {metrics["settling_time"]:.1f}ms, '
                                               f'Overshoot: {metrics["overshoot"]:.1f}%')
                                        ax.plot(t, resp, color=color, alpha=0.7, label=label)
                                        ax.legend(loc='upper right', fontsize='x-small', bbox_to_anchor=(0.98, 0.98))
                            else:
                                print("No valid responses found")
                        except Exception as e:
                            print(f"Error processing CSV file: {e}")
                except Exception as e:
                    print(f"Error processing log {log_index}: {e}")
            
            # Clean up any event files created in the logs directory
            for event_file in glob.glob(f'logs/{base_name}*.event'):
                try:
                    os.remove(event_file)
                    print(f"Removed: {event_file}")
                except Exception as e:
                    print(f"Could not remove {event_file}: {e}")
            
            color_idx += 1
    
    # Create warning text from actual settings
    if all_settings:
        # Use the settings from the first log (they should be the same for all logs)
        settings = all_settings[0]
        
        # Format the warning text with actual values
        warning_text = "IMPORTANT FACTORS AFFECTING STEP RESPONSE"
        
        # Format each section
        ff_section = f"Feedforward: {settings['ff_weight']} (weight) | {settings['feedforward_boost']} (boost) | {settings['feedforward_smooth_factor']} (smooth) | {settings['feedforward_jitter_factor']} (jitter) | {settings['feedforward_max_rate_limit']} (max rate)"
        
        filter_section = f"Filtering: {settings['dterm_lpf1_static_hz']}Hz (D term) | {settings['dterm_lpf2_static_hz']}Hz (D term 2) | {settings['gyro_lpf2_static_hz']}Hz (gyro) | {settings['dyn_notch_min_hz']}-{settings['dyn_notch_max_hz']}Hz (notch)"
        
        i_section = f"I term: {settings['iterm_windup']} (windup) | {settings['iterm_relax']} (relax) | {settings['iterm_relax_cutoff']}Hz (cutoff)"
        
        d_section = f"D term: {settings['d_min']} (min) | {settings['d_max_gain']} (max gain)"
        
        other_section = f"Other: {settings['anti_gravity_gain']} (anti gravity) | {settings['rc_smoothing_feedforward_cutoff']}Hz (RC smooth) | {settings['pidsum_limit']} (PID limit)"
    else:
        # Fallback if no settings are found
        warning_text = "WARNING: Step response is influenced by various Betaflight settings"
        ff_section = filter_section = i_section = d_section = other_section = ""
    
    # Add centered title
    fig.text(0.5, 0.94, warning_text, fontsize=12, fontweight='bold', ha='center')
    
    if all_settings:
        # Create a multi-line string with all sections
        all_text = (
            ff_section + "\n" +
            filter_section + "\n" +
            i_section + "\n" +
            d_section + "\n" +
            other_section
        )
        
        # Use a text box for the warning info, with explicit left alignment
        from matplotlib.offsetbox import AnchoredText
        from matplotlib.patheffects import withStroke
        
        # Create a text box with all content
        text_box = AnchoredText(all_text, 
                               loc='upper center',
                               pad=0.3,
                               bbox_to_anchor=(0.5, 0.925),
                               bbox_transform=fig.transFigure,
                               borderpad=0.3,
                               prop=dict(fontsize=9, ha='left'))
        
        # Make the text box completely transparent (no visible background or border)
        text_box.patch.set_boxstyle("square,pad=0.3")
        text_box.patch.set_facecolor('none')
        text_box.patch.set_alpha(1.0)
        text_box.patch.set_edgecolor('none')
        
        # Add the text box to the figure
        fig.add_artist(text_box)

    # Setup axes
    setup_axis(ax1, 'Roll')
    setup_axis(ax2, 'Pitch')
    setup_axis(ax3, 'Yaw')
    ax3.set_xlabel('Time (ms)')

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Run the analysis and generate plot
    plot_responses({}, 'step_responses.png')
    print("\nAnalysis complete. Results saved to step_responses.png")

if __name__ == "__main__":
    main() 