"""
AutoGuardian v2 — Multi-Signal RL Safety System
================================================
Features:
  - Multi-metric monitoring (loss, reward, entropy, KL, grad norm)
  - Adaptive sensitivity (tightens as training stabilizes)
  - Predictive collapse detection (gradient trend analysis)
  - Causal attribution (post-mortem reports on reverts)
  - Configurable per-metric thresholds
"""

import numpy as np
from collections import deque
import time


class MetricTracker:
    """Tracks a single metric with rolling statistics."""

    def __init__(self, name, window=50, sensitivity=2.0,
                 direction="lower_is_better"):
        self.name = name
        self.window = window
        self.base_sensitivity = sensitivity
        self.sensitivity = sensitivity
        self.direction = direction  # "lower_is_better" or "higher_is_better"

        self.buffer = deque(maxlen=window)
        self.full_history = []  # For trend analysis
        self.gradient_history = deque(maxlen=10)  # Rate of change

    def update(self, value):
        self.buffer.append(value)
        self.full_history.append(value)

        # Track rate of change (gradient)
        if len(self.full_history) >= 2:
            grad = value - self.full_history[-2]
            self.gradient_history.append(grad)

    @property
    def mean(self):
        return np.mean(self.buffer) if self.buffer else 0.0

    @property
    def std(self):
        return np.std(self.buffer) if len(self.buffer) > 1 else 1.0

    @property
    def threshold(self):
        """Dynamic threshold based on current sensitivity."""
        if self.direction == "lower_is_better":
            return self.mean * self.sensitivity
        else:
            return self.mean / self.sensitivity

    def is_anomalous(self, value):
        """Check if value exceeds the safety threshold."""
        if len(self.buffer) < 5:
            return False
        if self.direction == "lower_is_better":
            return value > self.threshold
        else:
            return value < self.threshold

    def trend_slope(self, window=5):
        """Linear regression slope over recent values. Positive = worsening for loss."""
        if len(self.full_history) < window:
            return 0.0
        recent = self.full_history[-window:]
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        return slope

    def predict_breach_in(self, n_steps=5):
        """
        Predict if metric will breach threshold within n_steps.
        Uses linear extrapolation of recent trend.
        Returns: estimated steps until breach, or None if safe.
        """
        if len(self.buffer) < 5:
            return None
        slope = self.trend_slope()
        if slope <= 0 and self.direction == "lower_is_better":
            return None  # Improving
        if slope >= 0 and self.direction == "higher_is_better":
            return None

        current = self.full_history[-1]
        threshold = self.threshold

        if self.direction == "lower_is_better":
            if slope <= 0:
                return None
            steps = (threshold - current) / slope
        else:
            if slope >= 0:
                return None
            steps = (current - threshold) / abs(slope)

        return int(steps) if 0 < steps <= n_steps else None

    def adapt_sensitivity(self, training_progress):
        """
        Tighten sensitivity as training stabilizes.
        training_progress: float 0.0 (start) to 1.0 (end)
        """
        # Start loose (base * 1.5), end tight (base * 0.75)
        scale = 1.5 - (0.75 * training_progress)
        self.sensitivity = self.base_sensitivity * scale


class RevertReport:
    """Causal attribution report for a revert event."""

    def __init__(self, step, trigger_metric, metrics_snapshot,
                 recent_observations=None):
        self.step = step
        self.timestamp = time.time()
        self.trigger_metric = trigger_metric
        self.metrics_snapshot = metrics_snapshot
        self.recent_observations = recent_observations or []
        self.diagnosis = self._diagnose()

    def _diagnose(self):
        """Auto-generate a diagnosis based on metric patterns."""
        diag = []
        snap = self.metrics_snapshot

        # Check for reward hacking (reward up but loss also up)
        if "reward" in snap and "loss" in snap:
            if snap["reward"]["trend"] > 0 and snap["loss"]["trend"] > 0:
                diag.append("REWARD_HACKING: Reward increasing alongside loss — "
                            "possible exploitation")

        # Check for entropy collapse
        if "entropy" in snap:
            if snap["entropy"]["value"] < 0.1:
                diag.append("ENTROPY_COLLAPSE: Policy has become near-deterministic")

        # Check for gradient explosion
        if "grad_norm" in snap:
            if snap["grad_norm"]["value"] > 100:
                diag.append("GRADIENT_EXPLOSION: Gradient norms are extreme")

        # Check for KL divergence spike
        if "kl_divergence" in snap:
            if snap["kl_divergence"]["value"] > 0.5:
                diag.append("KL_SPIKE: Policy has diverged significantly "
                            "from reference")

        if not diag:
            diag.append(f"GENERAL_DRIFT: {self.trigger_metric} exceeded threshold")

        return diag

    def __repr__(self):
        lines = [f"\n🔬 Revert Report — Step {self.step}"]
        lines.append(f"   Trigger: {self.trigger_metric}")
        lines.append(f"   Diagnosis:")
        for d in self.diagnosis:
            lines.append(f"     • {d}")
        lines.append(f"   Metric Snapshot:")
        for name, info in self.metrics_snapshot.items():
            lines.append(f"     {name}: {info['value']:.4f} "
                         f"(avg={info['mean']:.4f}, trend={info['trend']:+.4f})")
        return "\n".join(lines)


class AutoGuardian:
    """
    Multi-signal RL training guardian with adaptive sensitivity
    and predictive collapse detection.
    
    Usage:
        guardian = AutoGuardian(history, metrics_config={
            "loss":         {"sensitivity": 2.0, "direction": "lower_is_better"},
            "reward":       {"sensitivity": 2.0, "direction": "higher_is_better"},
            "entropy":      {"sensitivity": 3.0, "direction": "higher_is_better"},
            "grad_norm":    {"sensitivity": 2.5, "direction": "lower_is_better"},
            "kl_divergence":{"sensitivity": 2.0, "direction": "lower_is_better"},
        })
        
        status = guardian.step(step, {
            "loss": loss_val,
            "reward": ep_reward,
            "entropy": policy_entropy,
        })
    """

    def __init__(self, history, metrics_config=None, patience=3,
                 cooldown=10, enable_prediction=True,
                 prediction_horizon=5, total_steps=None):
        self.history = history
        self.patience = patience
        self.cooldown = cooldown
        self.enable_prediction = enable_prediction
        self.prediction_horizon = prediction_horizon
        self.total_steps = total_steps  # For adaptive sensitivity

        # --- Setup Metric Trackers ---
        if metrics_config is None:
            metrics_config = {
                "loss": {"sensitivity": 2.0, "direction": "lower_is_better"},
            }

        self.trackers = {}
        for name, cfg in metrics_config.items():
            self.trackers[name] = MetricTracker(
                name=name,
                sensitivity=cfg.get("sensitivity", 2.0),
                direction=cfg.get("direction", "lower_is_better"),
                window=cfg.get("window", 50),
            )

        # --- Internal State ---
        self.bad_steps_counter = 0
        self.cooldown_counter = 0
        self.total_reverts = 0
        self.revert_reports = []
        self._step_count = 0
        self._warmup_steps = 5
        self._recent_observations = deque(maxlen=20)

    def step(self, step_num, metrics):
        """
        Call every training step.
        
        Args:
            step_num: Current step
            metrics: dict of metric_name -> value.
                     At minimum must include the metrics defined in metrics_config.
                     Extra metrics are silently ignored.
        
        Returns:
            str: "safe", "warning", "reverted", "cooldown", "warmup",
                 or "pre_warning" (predictive)
        """
        if isinstance(metrics, (int, float)):
            metrics = {"loss": metrics}

        self._step_count += 1
        self._recent_observations.append({
            "step": step_num,
            "metrics": metrics.copy(),
        })

        # --- Adaptive Sensitivity ---
        if self.total_steps:
            progress = min(1.0, self._step_count / self.total_steps)
            for tracker in self.trackers.values():
                tracker.adapt_sensitivity(progress)

        # --- Cooldown ---
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return "cooldown"

        # --- Warmup ---
        if self._step_count <= self._warmup_steps:
            for name, val in metrics.items():
                if name in self.trackers:
                    self.trackers[name].update(val)
            self.history.commit(step_num, metrics, tag="warmup")
            return "warmup"

        # --- Update Trackers & Check Anomalies ---
        anomalies = []
        for name, val in metrics.items():
            if name not in self.trackers:
                continue
            tracker = self.trackers[name]
            if tracker.is_anomalous(val):
                anomalies.append(name)

        # --- Predictive Check ---
        pre_warnings = []
        if self.enable_prediction:
            for name, tracker in self.trackers.items():
                breach_in = tracker.predict_breach_in(self.prediction_horizon)
                if breach_in is not None:
                    pre_warnings.append((name, breach_in))

        # --- Decision Logic ---
        if anomalies:
            self.bad_steps_counter += 1
            trigger = anomalies[0]
            tracker = self.trackers[trigger]
            print(f"   ⚠️ {trigger}: {metrics.get(trigger, '?'):.4f} > "
                  f"threshold {tracker.threshold:.4f} "
                  f"({self.bad_steps_counter}/{self.patience})")

            if self.bad_steps_counter >= self.patience:
                return self._trigger_emergency(step_num, trigger, metrics)

            return "warning"

        elif pre_warnings:
            # Predictive early warning (don't count as bad step yet)
            for name, eta in pre_warnings:
                print(f"   🔮 Predicted: {name} may breach in ~{eta} steps")
            # Still update normally
            self.bad_steps_counter = 0
            for name, val in metrics.items():
                if name in self.trackers:
                    self.trackers[name].update(val)
            self.history.commit(step_num, metrics, tag="pre_warning")
            return "pre_warning"

        else:
            # All clear
            self.bad_steps_counter = 0
            for name, val in metrics.items():
                if name in self.trackers:
                    self.trackers[name].update(val)
            self.history.commit(step_num, metrics, tag="safe")
            return "safe"

    def _trigger_emergency(self, current_step, trigger_metric, current_metrics):
        """Revert model and generate causal attribution report."""
        print(f"\n🚨 CIRCUIT BREAKER at Step {current_step}! "
              f"Trigger: {trigger_metric}")

        # Build metrics snapshot for report
        snapshot = {}
        for name, tracker in self.trackers.items():
            snapshot[name] = {
                "value": current_metrics.get(name, 0.0),
                "mean": tracker.mean,
                "std": tracker.std,
                "trend": tracker.trend_slope(),
                "threshold": tracker.threshold,
            }

        # Generate report
        report = RevertReport(
            step=current_step,
            trigger_metric=trigger_metric,
            metrics_snapshot=snapshot,
            recent_observations=list(self._recent_observations),
        )
        self.revert_reports.append(report)
        print(report)

        # Revert
        steps_to_rewind = self.patience + 2
        success = self.history.checkout(relative_back=steps_to_rewind)

        if success:
            self.total_reverts += 1
            self.bad_steps_counter = 0
            self.cooldown_counter = self.cooldown

            # Clean poisoned values from buffers
            for tracker in self.trackers.values():
                for _ in range(min(len(tracker.buffer), self.patience)):
                    if tracker.buffer:
                        tracker.buffer.pop()

            return "reverted"
        else:
            print("❌ Revert failed — not enough history.")
            return "failed"

    def get_status_summary(self):
        """Current guardian state for dashboards."""
        return {
            "bad_steps": self.bad_steps_counter,
            "cooldown_remaining": self.cooldown_counter,
            "total_reverts": self.total_reverts,
            "trackers": {
                name: {
                    "mean": t.mean,
                    "std": t.std,
                    "sensitivity": t.sensitivity,
                    "threshold": t.threshold,
                    "trend": t.trend_slope(),
                }
                for name, t in self.trackers.items()
            },
        }