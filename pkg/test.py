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
        time.sleep(0.1) 
        
        # 1. Simulate Data
        inputs = torch.randn(1, 10)
        if 25 <= step <= 35: 
            targets = torch.randn(1, 1) * 50  # Poison
            is_poison = True
            print(f"Step {step} [POISON BATCH] ", end="")
        else:
            targets = torch.randn(1, 1)       # Healthy
            is_poison = False
            print(f"Step {step} [Normal Batch] ", end="")

        # 2. Forward Pass ONLY (Do not backward yet!)
        outputs = model(inputs)
        loss = nn.MSELoss()(outputs, targets)
        loss_val = loss.item()
        
        # 3. Ask Guardian: "Is it safe to proceed?"
        status = guardian.step(step, loss_val)
        
        # 4. Conditional Updates based on Guardian's verdict
        if status == "safe":
            # ✅ GREEN LIGHT: Safe to learn
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"✅ Loss: {loss_val:.2f}")

        elif status == "warning":
            # ⚠️ YELLOW LIGHT: Learn, but be careful
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Warning is printed inside guardian.step()

        elif status == "reverted":
            # 🛑 RED LIGHT: Do NOT learn. 
            # We just loaded a snapshot. If we learn now, we re-corrupt it.
            print(f"   🛑 Revert triggered. Skipping Step {step}.")

        elif status == "cooldown":
            # 🧊 BLUE LIGHT: Frozen. Do NOT learn.
            # We are waiting for the poison data to pass.
            print(f"   🧊 Cooldown (Frozen). Loss: {loss_val:.2f}")

    print(f"\n🏁 Training Complete. Total Auto-Reverts: {guardian.total_reverts}")

if __name__ == "__main__":
    test_model_guard()