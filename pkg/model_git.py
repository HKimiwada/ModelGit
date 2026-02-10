"""
ModelHistory v2 — Version Control for Neural Networks
=====================================================
Features:
  - Linear timeline + branching
  - Commit diffing (per-layer weight delta analysis)
  - Selective layer revert
  - Hierarchical checkpointing (fine / coarse / milestone)
  - Memory-efficient CPU offloading
"""

import torch
import copy
import time
import numpy as np
from collections import deque, defaultdict


class Commit:
    """A single snapshot of model state + metadata."""
    __slots__ = [
        'id', 'step', 'timestamp', 'metrics', 'tag',
        'state_dict', 'memory_size_mb', 'branch', 'parent_id'
    ]

    def __init__(self, step, metrics, state_dict, tag="training",
                 branch="main", parent_id=None):
        self.id = f"{branch}/{step}_{int(time.time() * 1000) % 100000}"
        self.step = step
        self.timestamp = time.time()
        self.metrics = metrics  # dict: {"loss": ..., "reward": ..., etc.}
        self.tag = tag
        self.branch = branch
        self.parent_id = parent_id
        self.state_dict = state_dict
        self.memory_size_mb = self._calc_size(state_dict)

    @staticmethod
    def _calc_size(state_dict):
        total = sum(t.numel() * t.element_size() for t in state_dict.values())
        return total / (1024 * 1024)

    def __repr__(self):
        return (f"Commit({self.id}, step={self.step}, "
                f"tag={self.tag}, {self.memory_size_mb:.2f}MB)")


class ModelHistory:
    """
    Git-like version control for PyTorch models.
    
    Supports:
      - commit / checkout / diff / blame
      - Named branches
      - Selective layer revert
      - Hierarchical checkpointing (fine/coarse/milestone tiers)
    """

    def __init__(self, model, max_fine_slots=100, max_coarse_slots=50,
                 coarse_interval=50):
        self.model = model
        self.current_branch = "main"

        # --- Hierarchical Storage ---
        # Fine: every commit (recent history, high granularity)
        self.fine_timeline = deque(maxlen=max_fine_slots)
        # Coarse: every N steps (longer history, lower granularity)
        self.coarse_timeline = deque(maxlen=max_coarse_slots)
        self.coarse_interval = coarse_interval
        # Milestones: manually tagged (never auto-evicted)
        self.milestones = {}

        # --- Branching ---
        # branch_name -> list of Commit
        self.branches = defaultdict(list)

        # Internal tracking
        self._commit_count = 0
        self._last_commit_id = None

    # ------------------------------------------------------------------
    #  COMMIT
    # ------------------------------------------------------------------
    def commit(self, step, metrics, tag="training", branch=None):
        """
        Snapshot current model weights.
        
        Args:
            step: Training step number
            metrics: dict of metric_name -> value (e.g. {"loss": 0.5, "reward": 12.3})
            tag: Descriptive tag ("safe", "warmup", "cooldown", "milestone", etc.)
            branch: Branch name (defaults to current_branch)
        
        Returns:
            commit_id (str)
        """
        if branch is None:
            branch = self.current_branch

        # CPU-offloaded deep copy
        cpu_state = {
            k: v.detach().cpu().clone()
            for k, v in self.model.state_dict().items()
        }

        commit = Commit(
            step=step,
            metrics=metrics if isinstance(metrics, dict) else {"loss": metrics},
            state_dict=cpu_state,
            tag=tag,
            branch=branch,
            parent_id=self._last_commit_id,
        )

        # Fine timeline (always)
        self.fine_timeline.append(commit)
        self.branches[branch].append(commit)

        # Coarse timeline (periodic)
        if self._commit_count % self.coarse_interval == 0:
            self.coarse_timeline.append(commit)

        # Milestone (if tagged)
        if tag == "milestone":
            self.milestones[commit.id] = commit

        self._commit_count += 1
        self._last_commit_id = commit.id
        return commit.id

    def tag_milestone(self, name=None):
        """Tag the latest commit as a milestone."""
        if not self.fine_timeline:
            return None
        latest = self.fine_timeline[-1]
        key = name or latest.id
        latest.tag = "milestone"
        self.milestones[key] = latest
        return key

    # ------------------------------------------------------------------
    #  CHECKOUT
    # ------------------------------------------------------------------
    def checkout(self, step=None, commit_id=None, relative_back=None,
                 branch=None, layers=None):
        """
        Restore model to a previous state.
        
        Args:
            step: Specific step number to revert to
            commit_id: Specific commit ID
            relative_back: N steps back from latest
            branch: Search within specific branch
            layers: List of layer name prefixes for SELECTIVE revert.
                    e.g., layers=["value_head"] only reverts value head weights.
                    If None, reverts entire model.
        
        Returns:
            True if successful, False otherwise
        """
        target = self._find_commit(step, commit_id, relative_back, branch)
        if target is None:
            print("❌ Commit not found.")
            return False

        if layers is not None:
            # Selective layer revert
            current_state = self.model.state_dict()
            for key in target.state_dict:
                if any(key.startswith(prefix) for prefix in layers):
                    current_state[key] = target.state_dict[key]
            self.model.load_state_dict(current_state)
            print(f"✅ Selectively reverted layers {layers} to Step {target.step}")
        else:
            # Full revert
            self.model.load_state_dict(target.state_dict)
            print(f"✅ Reverted to Step {target.step} "
                  f"(Metrics: {target.metrics})")

        return True

    def _find_commit(self, step=None, commit_id=None, relative_back=None,
                     branch=None):
        """Search for a commit across timelines."""
        timeline = self.fine_timeline
        if branch and branch in self.branches:
            timeline = self.branches[branch]

        if relative_back is not None:
            idx = -(relative_back + 1)
            if abs(idx) > len(timeline):
                print(f"⚠️ Cannot go back {relative_back} steps "
                      f"(history: {len(timeline)})")
                return None
            return timeline[idx]

        if commit_id is not None:
            # Check milestones first
            if commit_id in self.milestones:
                return self.milestones[commit_id]
            for c in reversed(timeline):
                if c.id == commit_id:
                    return c

        if step is not None:
            # Search fine -> coarse -> milestones
            for c in reversed(timeline):
                if c.step == step:
                    return c
            for c in reversed(self.coarse_timeline):
                if c.step == step:
                    return c

        return None

    # ------------------------------------------------------------------
    #  DIFF
    # ------------------------------------------------------------------
    def diff(self, commit_a_step, commit_b_step, branch=None):
        """
        Compare two commits. Returns per-layer L2 distance + % change.
        Useful for diagnosing which part of the network collapsed.
        
        Returns:
            dict: {layer_name: {"l2_dist": float, "pct_change": float, "max_delta": float}}
        """
        a = self._find_commit(step=commit_a_step, branch=branch)
        b = self._find_commit(step=commit_b_step, branch=branch)
        if a is None or b is None:
            print("❌ Could not find one or both commits for diff.")
            return {}

        result = {}
        for key in a.state_dict:
            wa = a.state_dict[key].float()
            wb = b.state_dict[key].float()
            delta = wb - wa
            l2 = torch.norm(delta).item()
            base_norm = torch.norm(wa).item()
            pct = (l2 / base_norm * 100) if base_norm > 0 else float('inf')
            result[key] = {
                "l2_dist": l2,
                "pct_change": pct,
                "max_delta": torch.max(torch.abs(delta)).item(),
                "mean_delta": torch.mean(torch.abs(delta)).item(),
            }

        return result

    def print_diff(self, commit_a_step, commit_b_step, branch=None):
        """Pretty-print a diff."""
        diffs = self.diff(commit_a_step, commit_b_step, branch)
        if not diffs:
            return
        print(f"\n📊 Weight Diff: Step {commit_a_step} → Step {commit_b_step}")
        print("-" * 70)
        print(f"{'Layer':<35} {'L2 Dist':>10} {'% Change':>10} {'Max Δ':>10}")
        print("-" * 70)
        for layer, stats in sorted(diffs.items(),
                                     key=lambda x: x[1]['pct_change'],
                                     reverse=True):
            print(f"{layer:<35} {stats['l2_dist']:>10.4f} "
                  f"{stats['pct_change']:>9.2f}% {stats['max_delta']:>10.4f}")

    # ------------------------------------------------------------------
    #  BLAME
    # ------------------------------------------------------------------
    def blame(self, metric_name="loss", threshold=0.5):
        """Find the first commit where a metric exceeded threshold."""
        print(f"\n🔍 Blame Analysis: '{metric_name}' > {threshold}")
        for commit in self.fine_timeline:
            val = commit.metrics.get(metric_name, None)
            if val is not None and val > threshold:
                print(f"   ⚠️ Drift at Step {commit.step} "
                      f"({metric_name}={val:.4f})")
                return commit.step
        print("   No anomalies found.")
        return None

    # ------------------------------------------------------------------
    #  BRANCH MANAGEMENT
    # ------------------------------------------------------------------
    def create_branch(self, name):
        """Create a new branch from the current state."""
        if self.fine_timeline:
            latest = self.fine_timeline[-1]
            # Copy the latest commit as the branch root
            self.branches[name].append(latest)
        self.current_branch = name
        print(f"🌿 Created branch '{name}' from step {latest.step}")
        return name

    def switch_branch(self, name):
        """Switch to an existing branch and load its latest state."""
        if name not in self.branches or not self.branches[name]:
            print(f"❌ Branch '{name}' not found or empty.")
            return False
        self.current_branch = name
        latest = self.branches[name][-1]
        self.model.load_state_dict(latest.state_dict)
        print(f"🔀 Switched to branch '{name}' (step {latest.step})")
        return True

    def list_branches(self):
        """Show all branches and their commit counts."""
        for name, commits in self.branches.items():
            marker = " ← HEAD" if name == self.current_branch else ""
            print(f"  {'*' if marker else ' '} {name}: "
                  f"{len(commits)} commits{marker}")

    # ------------------------------------------------------------------
    #  UTILITIES
    # ------------------------------------------------------------------
    @property
    def timeline(self):
        """Backward compat with v1."""
        return self.fine_timeline

    def memory_usage_mb(self):
        """Total RAM usage across all tiers."""
        total = sum(c.memory_size_mb for c in self.fine_timeline)
        total += sum(c.memory_size_mb for c in self.coarse_timeline)
        total += sum(c.memory_size_mb for c in self.milestones.values())
        return total

    def summary(self):
        """Print a summary of the history state."""
        print(f"\n📦 ModelHistory Summary")
        print(f"   Fine commits:   {len(self.fine_timeline)}")
        print(f"   Coarse commits: {len(self.coarse_timeline)}")
        print(f"   Milestones:     {len(self.milestones)}")
        print(f"   Branches:       {list(self.branches.keys())}")
        print(f"   Memory usage:   {self.memory_usage_mb():.2f} MB")