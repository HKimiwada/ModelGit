"""
Bridge — File-based IPC between training loop and dashboard.
"""

import json
import os
import time

STATE_FILE = "monitor_state.json"
COMMAND_FILE = "monitor_commands.txt"
LOG_FILE = "training_log.csv"


def init_log():
    with open(LOG_FILE, "w") as f:
        f.write("step,loss,reward,entropy,grad_norm,status\n")


def write_state(step, metrics, status, history_len):
    data = {
        "step": step,
        "metrics": metrics,
        "status": status,
        "history_len": history_len,
        "timestamp": time.time(),
    }
    temp = STATE_FILE + ".tmp"
    with open(temp, "w") as f:
        json.dump(data, f)
    os.replace(temp, STATE_FILE)


def append_history(step, metrics, status):
    loss = metrics.get("loss", 0)
    reward = metrics.get("reward", 0)
    entropy = metrics.get("entropy", 0)
    grad_norm = metrics.get("grad_norm", 0)
    with open(LOG_FILE, "a") as f:
        f.write(f"{step},{loss},{reward},{entropy},{grad_norm},{status}\n")


def check_commands():
    if os.path.exists(COMMAND_FILE):
        with open(COMMAND_FILE, "r") as f:
            cmd = f.read().strip()
        os.remove(COMMAND_FILE)
        return cmd
    return None