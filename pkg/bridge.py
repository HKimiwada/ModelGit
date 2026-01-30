import json
import os
import time

STATE_FILE = "monitor_state.json"
COMMAND_FILE = "monitor_commands.txt"
LOG_FILE = "training_log.csv"

def init_log():
    # Reset the log file at start of training
    with open(LOG_FILE, "w") as f:
        f.write("step,loss,status\n")

def write_state(step, loss, status, history_len):
    """
    Saves the 'heartbeat' for the dashboard to read.
    Uses atomic writing (write + rename) to prevent reading crashes.
    """
    data = {
        "step": step,
        "loss": loss,
        "status": status, 
        "history_len": history_len,
        "timestamp": time.time()
    }
    temp_file = STATE_FILE + ".tmp"
    with open(temp_file, "w") as f:
        json.dump(data, f)
    os.replace(temp_file, STATE_FILE)

def append_history(step, loss, status):
    """
    Appends to the CSV so we can plot the full line chart.
    """
    with open(LOG_FILE, "a") as f:
        f.write(f"{step},{loss},{status}\n")

def check_commands():
    """
    Checks if the dashboard button was pressed.
    Returns: 'POISON_ON', 'REVERT_NOW', or None
    """
    if os.path.exists(COMMAND_FILE):
        with open(COMMAND_FILE, "r") as f:
            cmd = f.read().strip()
        os.remove(COMMAND_FILE)
        return cmd
    return None