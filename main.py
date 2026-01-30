# Testing Script
# test_timemachine.py
import torch
import torch.nn as nn
import random
import time
from model_git import ModelHistory
from model_guard import AutoGuardian # Assuming you saved the class above

def test_model_git():
    """
    Test the ModelHistory (Time Machine) functionality.
    Simulates a training loop with a sudden drift in loss,
    then uses ModelHistory to revert to a previous safe state.
    """
    # 1. Setup a dummy model
    model = nn.Linear(10, 1) # Simple 1-layer model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    history = ModelHistory(model, max_memory_slots=50)

    print("🚀 Starting Training Simulation...")

    # 2. Simulate Training Loop
    for step in range(20):
        # Fake Loss: Starts high, goes low... then spikes (Drift!)
        if step < 15:
            loss_val = 1.0 - (step * 0.05) # Good training
        else:
            loss_val = 5.0 + (step * 0.1)  # CATASTROPHIC DRIFT (Steps 15-19)
        
        # Standard PyTorch stuff
        loss = torch.tensor([loss_val], requires_grad=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # --- MODELGIT INTEGRATION ---
        history.commit(step=step, metric=loss_val)
        print(f"Step {step}: Loss {loss_val:.2f} (Committed)")

    # 3. Simulate the Panic Button
    print("\n🚨 ALERT: Model has drifted! Current Loss is HIGH.")
    print("Current Weight Sample:", model.weight.data[0][0].item())

    # 4. The Magic Revert
    # Let's go back to Step 14 (right before the drift started)
    print("\n⏪ Initiating Time Machine...")
    history.checkout(step=14)

    print("\n✅ Model Restored.")
    print("Restored Weight Sample:", model.weight.data[0][0].item())

def test_model_guard():
    # --- SETUP ---
    model = nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Initialize our System
    history = ModelHistory(model)
    guardian = AutoGuardian(history, sensitivity=1.5, patience=2)

    print("Guardian Online. Training Monitor Active.\n")

    # --- SIMULATION LOOP ---
    for step in range(50):
        time.sleep(0.1) # Just to make the demo readable
        
        # 1. Simulate Data
        # Normal data: Input 1.0 -> Target 1.0
        # Poison data: Input 1.0 -> Target -10.0 (Will cause huge error)
        inputs = torch.randn(1, 10)
        
        if 25 <= step <= 35: 
            # INJECT POISON between steps 25 and 35
            # This simulates "Concept Drift" or corrupted sensors
            targets = torch.randn(1, 1) * 50 
            is_poison = True
            print(f"Step {step} [POISON BATCH] ", end="")
        else:
            # Healthy Data
            targets = torch.randn(1, 1)
            is_poison = False
            print(f"Step {step} [Normal Batch] ", end="")

        # 2. Standard Training Step
        outputs = model(inputs)
        loss = nn.MSELoss()(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        
        # 3. THE GUARDIAN INTERVENTION
        status = guardian.step(step, loss_val)
        
        # Visual Feedback
        if status == "safe":
            print(f"✅ Loss: {loss_val:.2f}")
        elif status == "warning":
            # Warning is printed inside class
            pass
        elif status == "reverted":
            print("   Testing fix... (skipping this batch)")
            # In a real loop, you might also skip the next data loader item
        elif status == "cooldown":
            print(f"🧊 Cooldown Mode. Loss: {loss_val:.2f}")

    print(f"\n🏁 Training Complete. Total Auto-Reverts: {guardian.total_reverts}")

if __name__ == "__main__":
    test_model_guard()