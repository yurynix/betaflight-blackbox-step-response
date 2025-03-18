# Betaflight Step Response Analyzer

This tool analyzes step responses from Betaflight blackbox logs, providing insights into the flight controller's PID performance through time-domain analysis.

## Features

- Processes multiple blackbox logs simultaneously
- Analyzes Roll, Pitch, and Yaw axes
- Calculates and displays key metrics:
  - Rise time (90% of steady state)
  - Settling time (within 2% of steady state)
  - Overshoot percentage
- Generates high-quality plots with averaged step responses

## Prerequisites

- Python 3.8 or higher
- Betaflight Blackbox Tools

### Installing Blackbox Tools

1. Clone the Betaflight Blackbox Tools repository:
```bash
git clone https://github.com/betaflight/blackbox-tools.git
cd blackbox-tools
```

2. Build the tools:
```bash
make
```

3. Copy the `blackbox_decode` executable to this project's directory:
```bash
cp blackbox_decode /path/to/this/project/
```

## Installation

1. Clone this repository:
```bash
git clone git@github.com:yurynix/betaflight-blackbox-step-response.git
cd betaflight-blackbox-step-response
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your Betaflight blackbox logs (`.bbl` files) in the `logs` directory.

2. Run the analysis script:
```bash
python plot_step_response.py
```

3. The script will:
   - Convert BBL files to CSV format using blackbox_decode
   - Process each log file
   - Generate plots showing step responses for all axes
   - Save the results as `step_responses.png`

## Output

The generated plot (`step_responses.png`) shows:
- Averaged step responses for each axis (Roll, Pitch, Yaw)
- Multiple log files plotted in different colors
- Key metrics displayed for each response
- Reference levels and timing markers

## Notes

- The script automatically handles different log rates and file durations
- Step detection uses adaptive thresholds and validation criteria
- Responses are normalized and averaged for better comparison
- Temporary CSV files are automatically cleaned up after processing 