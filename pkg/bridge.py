import json
import os
import time

# Shared paths (files created in Project Root)
STATE_FILE = "monitor_state.json"
COMMAND_FILE = "monitor_commands.txt"
LOG_FILE = "training_log.csv"

def init_log():
    with open(LOG_FILE, "w") as f:
        f.write("step,loss,status\n")

def write_state(step, loss, status, history_len):
    data = {
        "step": step,
        "loss": loss,
        "status": status,
        "history_len": history_len,
        "timestamp": time.time()
    }
    # Atomic write to prevent read errors
    temp = STATE_FILE + ".tmp"
    with open(temp, "w") as f:
        json.dump(data, f)
    os.replace(temp, STATE_FILE)

def append_history(step, loss, status):
    with open(LOG_FILE, "a") as f:
        f.write(f"{step},{loss},{status}\n")

def check_commands():
    if os.path.exists(COMMAND_FILE):
        with open(COMMAND_FILE, "r") as f:
            cmd = f.read().strip()
        os.remove(COMMAND_FILE)
        return cmd
    return None