import torch
import torch.nn as nn
import time
import random
import sys

# Import from our custom package
from pkg import ModelHistory, AutoGuardian, bridge

def run_training_demo():
    # --- SETUP ---
    model = nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Initialize ModelGit System
    history = ModelHistory(model, max_memory_slots=100)
    # Sensitivity 2.0 = Panic if loss is 2x higher than average
    guardian = AutoGuardian(history, sensitivity=2.0, patience=2, cooldown=5)

    # Initialize Dashboard Logs
    bridge.init_log()
    print("🛡️  Guardian Online. Dashboard Bridge Active.")
    print("   Run 'streamlit run dashboard.py' in another terminal to control this.")

    # State flags for the demo
    manual_poison = False
    
    # --- LOOP ---
    for step in range(500): # Long loop for demo purposes
        time.sleep(0.3) # Slow down slightly so we can watch the dashboard
        
        # 1. Check for Dashboard Commands
        cmd = bridge.check_commands()
        if cmd == "POISON_ON":
            manual_poison = True
            print(f"\n[COMMAND] ☣️  Received POISON_ON signal!")
        elif cmd == "REVERT_NOW":
            print(f"\n[COMMAND] ⏪ Received REVERT_NOW signal!")
            guardian._trigger_emergency_protocol(step)

        # 2. Generate Data (Simulate Environment)
        inputs = torch.randn(1, 10)
        
        # Poison Condition: 
        # Triggered by (Manual Button) OR (Hardcoded Steps 50-70)
        is_poison_active = manual_poison or (50 <= step <= 70)
        
        if is_poison_active:
            # POISON DATA: Target is wildly wrong (-50.0)
            targets = torch.randn(1, 1) + 50.0
            print(f"Step {step} [☣️ POISON] ", end="")
            # Auto-disable manual poison after 20 steps to let it heal
            if manual_poison and random.random() < 0.05: 
                manual_poison = False
                print("(Poison supply exhausted) ", end="")
        else:
            # NORMAL DATA: Target is normal distribution
            targets = torch.randn(1, 1)
            print(f"Step {step} [Normal]    ", end="")

        # 3. Forward Pass
        outputs = model(inputs)
        loss = nn.MSELoss()(outputs, targets)
        loss_val = loss.item()

        # 4. Guardian Safety Check
        status = guardian.step(step, loss_val)

        # 5. Conditional Update Logic
        if status == "safe":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"✅ Loss: {loss_val:.4f}")
            
        elif status == "warning":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # (Warning printed by guardian internal logic)

        elif status == "reverted":
            # 🛑 STOP! Do not learn.
            print(f"🛑 REVERTING! Skipping update.")
            
        elif status == "cooldown":
            # 🧊 FREEZE! Do not learn.
            print(f"🧊 Cooldown. Loss: {loss_val:.4f} (Ignored)")

        # 6. Write to Bridge (For Dashboard)
        bridge.write_state(step, loss_val, status, len(history.timeline))
        bridge.append_history(step, loss_val, status)

if __name__ == "__main__":
    try:
        run_training_demo()
    except KeyboardInterrupt:
        print("\nDemo stopped.")