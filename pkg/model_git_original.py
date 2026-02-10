"""
Build the ModelHistory class that wraps a PyTorch model and manages an in-memory "undo" stack.

"""
import torch
import copy
import time
from collections import deque
import sys

class ModelHistory:
    def __init__(self, model, max_memory_slots=100):
        """
        Args:
            model: The PyTorch model to track.
            max_memory_slots: How many past versions to keep in RAM.
                              (Older versions are discarded automatically).
        """
        self.model = model
        self.max_memory_slots = max_memory_slots
        
        # We use a deque (double-ended queue) for O(1) appends and pops.
        # This is our 'Timeline'.
        self.timeline = deque(maxlen=max_memory_slots)
        
    def commit(self, step, metric, tag="training"):
        """
        Takes a snapshot of the current model state.
        CRITICAL: Offloads to CPU to prevent GPU OOM.
        """
        # 1. Capture the State Dict
        # We must DETACH (remove gradients), move to CPU, and CLONE (deep copy).
        # If we don't clone, we just save a pointer to the changing weights.
        cpu_state_dict = {
            k: v.detach().cpu().clone() 
            for k, v in self.model.state_dict().items()
        }
        
        # 2. Create the Commit Object
        commit_data = {
            'id': f"{step}_{int(time.time())}", # Unique ID
            'step': step,
            'timestamp': time.time(),
            'metric': metric,  # e.g., Loss or Accuracy
            'tag': tag,
            'state_dict': cpu_state_dict,
            'memory_size_mb': self._get_size_mb(cpu_state_dict)
        }
        
        self.timeline.append(commit_data)
        return commit_data['id']

    def checkout(self, step=None, commit_id=None, relative_back=None):
        """
        Restores the model to a previous state.
        Can lookup by specific step, ID, or "N steps ago".
        """
        target_commit = None
        
        # Option 1: Go back N steps (e.g., "undo last 5 updates")
        if relative_back is not None:
            if relative_back >= len(self.timeline):
                print(f"⚠️ Error: Cannot go back {relative_back} steps. History only has {len(self.timeline)}.")
                return False
            # Index -1 is current, -2 is one step back, etc.
            target_commit = self.timeline[-(relative_back + 1)]

        # Option 2: Find by specific Step Number
        elif step is not None:
            # Search backwards (assuming recent is more likely)
            for commit in reversed(self.timeline):
                if commit['step'] == step:
                    target_commit = commit
                    break
        
        if target_commit:
            # Load the CPU weights back into the (potentially GPU) model
            self.model.load_state_dict(target_commit['state_dict'])
            print(f"✅ Reverted to Step {target_commit['step']} (Metric: {target_commit['metric']:.4f})")
            return True
        else:
            print("❌ Commit not found.")
            return False

    def blame(self, threshold=0.5):
        """
        Debugging Tool: Finds the first commit where metrics dropped below threshold.
        """
        print(f"\n🔍 Running Blame Analysis (Threshold: {threshold})...")
        for i, commit in enumerate(self.timeline):
            if commit['metric'] > threshold: # Assuming 'metric' is loss (lower is better)
                print(f"   -> ⚠️ Drift detected at Step {commit['step']} (Loss: {commit['metric']:.4f})")
                return commit['step']
        print("   -> No anomalies found in current history.")
        return None

    def _get_size_mb(self, state_dict):
        """Helper to track how much RAM we are eating."""
        total_bytes = 0
        for t in state_dict.values():
            total_bytes += t.numel() * t.element_size()
        return total_bytes / (1024 * 1024)