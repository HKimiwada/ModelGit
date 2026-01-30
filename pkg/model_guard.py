import numpy as np
from collections import deque

class AutoGuardian:
    def __init__(self, history, sensitivity=2.0, patience=3, cooldown=10):
        """
        Args:
            history: Instance of ModelHistory (the Time Machine).
            sensitivity: How many times worse than average is allowed? 
                         (e.g., 2.0 means if loss doubles, we panic).
            patience: How many bad steps to tolerate before reverting?
            cooldown: Steps to wait after a revert before checking again 
                      (prevents infinite revert loops).
        """
        self.history = history
        self.sensitivity = sensitivity
        self.patience = patience
        self.cooldown = cooldown
        
        # Internal State
        self.loss_buffer = deque(maxlen=50) # To calculate "Normal" behavior
        self.bad_steps_counter = 0
        self.cooldown_counter = 0
        self.total_reverts = 0

    def step(self, step_num, current_loss):
        """
        Call this every training step.
        Returns: 'continue', 'warning', or 'reverted'
        """
        # 1. Handle Cooldown (Don't panic immediately after fixing)
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            # Still record history, but don't check for drift
            # self.history.commit(step_num, current_loss, tag="cooldown")
            return "cooldown"

        # 2. Calculate Baseline (Moving Average of last 50 steps)
        if len(self.loss_buffer) < 5:
            # Not enough data yet to judge safety
            self.loss_buffer.append(current_loss)
            self.history.commit(step_num, current_loss, tag="warmup")
            return "warmup"

        avg_loss = np.mean(self.loss_buffer)
        safety_threshold = avg_loss * self.sensitivity

        # 3. The Safety Check
        if current_loss > safety_threshold:
            self.bad_steps_counter += 1
            print(f"   ⚠️ Warning: Loss {current_loss:.4f} > Threshold {safety_threshold:.4f} ({self.bad_steps_counter}/{self.patience})")
            
            if self.bad_steps_counter >= self.patience:
                return self._trigger_emergency_protocol(step_num)
            
            return "warning"
        
        else:
            # All good! Reset counters and update baseline
            self.bad_steps_counter = 0
            self.loss_buffer.append(current_loss)
            self.history.commit(step_num, current_loss, tag="safe")
            return "safe"

    def _trigger_emergency_protocol(self, current_step):
        """
        The Nuclear Option: Reverts the model.
        """
        print(f"\n🚨 CIRCUIT BREAKER TRIPPED at Step {current_step}!")
        
        # Go back to where we were safe (patience + 1 steps ago)
        # If patience is 3, we want to go back 4 steps to be sure.
        steps_to_rewind = self.patience + 2 
        
        success = self.history.checkout(relative_back=steps_to_rewind)
        
        if success:
            self.total_reverts += 1
            self.bad_steps_counter = 0
            self.cooldown_counter = self.cooldown # Give it time to stabilize
            
            # CRITICAL: We must clean our baseline buffer. 
            # If we keep the high loss values in the buffer, the average goes up,
            # and the model accepts bad behavior as the "new normal".
            # So, we remove the recent "poisoned" values.
            for _ in range(min(len(self.loss_buffer), self.patience)):
                self.loss_buffer.pop()
                
            return "reverted"
        else:
            return "failed"