"""
Interactive Training Demo — works with Streamlit dashboard.
Run this in one terminal, `streamlit run dashboard.py` in another.
"""

import torch
import torch.nn as nn
import time
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pkg import ModelHistory, AutoGuardian, bridge


def run_training_demo():
    # --- SETUP ---
    model = nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    history = ModelHistory(model, max_fine_slots=100)
    guardian = AutoGuardian(
        history,
        metrics_config={
            "loss": {"sensitivity": 2.0, "direction": "lower_is_better"},
        },
        patience=2,
        cooldown=5,
        total_steps=500,
    )

    bridge.init_log()
    print("🛡️  Sentinel Guardian Online. Dashboard Bridge Active.")
    print("   Run 'streamlit run dashboard.py' in another terminal.")

    manual_poison = False

    for step in range(500):
        time.sleep(0.3)

        # 1. Dashboard Commands
        cmd = bridge.check_commands()
        if cmd == "POISON_ON":
            manual_poison = True
            print(f"\n[COMMAND] ☣️  Received POISON_ON signal!")
        elif cmd == "REVERT_NOW":
            print(f"\n[COMMAND] ⏪ Received REVERT_NOW signal!")
            guardian._trigger_emergency(step, "manual", {"loss": 999})

        # 2. Generate Data
        inputs = torch.randn(1, 10)
        is_poison_active = manual_poison or (50 <= step <= 70)

        if is_poison_active:
            targets = torch.randn(1, 1) + 50.0
            print(f"Step {step} [☣️ POISON] ", end="")
            if manual_poison and random.random() < 0.05:
                manual_poison = False
                print("(Poison exhausted) ", end="")
        else:
            targets = torch.randn(1, 1)
            print(f"Step {step} [Normal]    ", end="")

        # 3. Forward Pass
        outputs = model(inputs)
        loss = nn.MSELoss()(outputs, targets)
        loss_val = loss.item()

        # Compute gradient norm
        optimizer.zero_grad()
        loss.backward()
        grad_norm = sum(
            p.grad.data.norm(2).item() ** 2
            for p in model.parameters() if p.grad is not None
        ) ** 0.5

        metrics = {"loss": loss_val, "grad_norm": grad_norm}

        # 4. Guardian Check
        status = guardian.step(step, metrics)

        # 5. Conditional Update
        if status in ("safe", "warning", "warmup", "pre_warning"):
            optimizer.step()
            print(f"✅ Loss: {loss_val:.4f}")
        elif status == "reverted":
            print(f"🛑 REVERTING! Skipping update.")
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        elif status == "cooldown":
            print(f"🧊 Cooldown. Loss: {loss_val:.4f} (Ignored)")

        # 6. Bridge
        bridge.write_state(step, metrics, status, len(history.fine_timeline))
        bridge.append_history(step, metrics, status)


if __name__ == "__main__":
    try:
        run_training_demo()
    except KeyboardInterrupt:
        print("\nDemo stopped.")