# SPDX-License-Identifier: Apache-2.0

"""Capacity manager for rollout generation.

This module provides the StalenessManager class which manages rollout
concurrency for asynchronous rollout generation in RL training. Version
freshness is checked by the workflow executor using the per-task rollout
version.
"""

from threading import Lock
from typing import Protocol

from areal.api import RolloutStat


class VersionProvider(Protocol):
    def get_version(self) -> int:
        raise NotImplementedError()


class StalenessManager:
    """Manages rollout capacity based on concurrency constraints.

    The manager ensures that the number of concurrent rollouts does not exceed
    the configured maximum.

    Parameters
    ----------
    version_provider : VersionProvider
        Provider for current model version (e.g., InferenceEngine)
    max_concurrent_rollouts : int
        Maximum number of concurrent rollouts allowed
    consumer_batch_size : int
        Expected batch size for consuming rollouts during training
    max_staleness : int
        Maximum allowed offpolicyness (version difference) for rollouts
    """

    def __init__(
        self,
        version_provider: VersionProvider,
        max_concurrent_rollouts: int,
        consumer_batch_size: int,
        max_staleness: int,
    ):
        """Initialize the staleness manager.

        Parameters
        ----------
        version_provider : VersionProvider
            Provider for current model version (e.g., InferenceEngine)
        max_concurrent_rollouts : int
            Maximum number of concurrent rollouts allowed
        consumer_batch_size : int
            Expected batch size for consuming rollouts during training
        max_staleness : int
            Maximum allowed offpolicyness (version difference) for rollouts
        """
        self.version_provider = version_provider
        self.max_concurrent_rollouts = max_concurrent_rollouts
        self.consumer_batch_size = consumer_batch_size
        self.max_staleness = max_staleness

        # Thread-safe access to rollout statistics
        self.lock = Lock()
        self.rollout_stat = RolloutStat()

    def get_pending_limit(self) -> int:
        """Get the maximum number of pending rollouts allowed.

        Returns
        -------
        int
            Maximum number of pending rollouts (enqueued).
        """
        return max(1, self.max_concurrent_rollouts)

    def get_capacity(self) -> int:
        """Calculate available capacity for new rollouts.

        Returns
        -------
        int
            Number of new rollout slots available. Can be negative if over capacity.
        """
        with self.lock:
            max_concurrent_rollouts = max(1, self.max_concurrent_rollouts)
            return max_concurrent_rollouts - self.rollout_stat.running

    def on_rollout_enqueued(self) -> None:
        """Callback when a rollout is enqueued as a pending input task.

        Thread-safe method to increment the enqueued counters.
        """
        with self.lock:
            self.rollout_stat.enqueued += 1

    def on_rollout_submitted(self) -> None:
        """Callback when a rollout is submitted for execution.

        Thread-safe method to decrement enqueued counter and increment running counters.
        """
        with self.lock:
            self.rollout_stat.enqueued -= 1
            self.rollout_stat.running += 1

    def on_rollout_accepted(self) -> None:
        """Callback when a rollout completes successfully and is accepted.

        Thread-safe method to decrement running counter and increment accepted counter.
        """
        with self.lock:
            self.rollout_stat.running -= 1
            self.rollout_stat.accepted += 1

    def on_rollout_rejected(self) -> None:
        """Callback when a rollout completes but is rejected.

        Thread-safe method to decrement running counter and increment rejected counter.
        """
        with self.lock:
            self.rollout_stat.running -= 1
            self.rollout_stat.rejected += 1

    def on_rollout_invalidated(self) -> None:
        """Move a completed rollout from accepted to rejected.

        This is used when a result was fresh at completion but became stale while
        waiting in a result buffer. The rollout is no longer running, so this must
        not change the running counter.
        """
        with self.lock:
            if self.rollout_stat.accepted <= 0:
                raise RuntimeError("Cannot invalidate a rollout that was not accepted.")
            self.rollout_stat.accepted -= 1
            self.rollout_stat.rejected += 1

    def get_stats(self) -> RolloutStat:
        """Get a snapshot of current rollout statistics.

        Returns
        -------
        RolloutStat
            Current rollout statistics (enqueued, accepted, running)
        """
        with self.lock:
            return RolloutStat(
                accepted=self.rollout_stat.accepted,
                enqueued=self.rollout_stat.enqueued,
                rejected=self.rollout_stat.rejected,
                running=self.rollout_stat.running,
            )
