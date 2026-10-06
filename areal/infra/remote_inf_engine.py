# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
import random
import re
import shutil
import subprocess
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import Future
from contextlib import contextmanager, nullcontext
from datetime import datetime
from logging import Logger
from threading import Condition, Lock
from typing import TYPE_CHECKING, Any, Protocol

import aiohttp
import numpy as np
import ray
import requests
import torch.distributed as dist
import uvloop
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api import (
    InferenceEngine,
    LocalInfServerInfo,
    ModelRequest,
    ModelResponse,
    ParamSpec,
    RolloutWorkflow,
    WeightUpdateMeta,
    WorkflowLike,
)
from areal.api.cli_args import InferenceEngineConfig, OpenAIProxyConfig
from areal.api.io_struct import (
    HttpGenerationResult,
    HttpRequest,
    WeightUpdateRequests,
)
from areal.infra import workflow_context
from areal.infra.platforms import current_platform
from areal.infra.utils.concurrent import get_executor
from areal.infra.utils.http import arequest_with_retry, get_default_connector
from areal.infra.utils.launcher import wait_llm_server_addrs
from areal.infra.utils.proc import kill_process_tree
from areal.utils import logging, name_resolve, names
from areal.utils.data import concat_padded_tensors
from areal.utils.dynamic_import import import_from_string
from areal.utils.network import (
    find_free_ports,
    format_hostport,
    gethostip,
    split_hostport,
)
from areal.utils.perf_tracer import trace_perf

from .workflow_executor import RolloutStaleError, WorkflowExecutor

if TYPE_CHECKING:
    from areal.experimental.openai import InteractionWithTokenLogpReward

RID_CACHE_SIZE = 128
_REQUEST_MAX_ATTEMPTS_METADATA_KEY = "_request_max_attempts"

logger = logging.getLogger("RemoteInfEngine")


class GroupedRolloutWorkflow(RolloutWorkflow):
    def __init__(
        self,
        workflow: RolloutWorkflow,
        group_size: int,
        logger: Logger,
    ):
        if group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {group_size}")
        self.workflow = workflow
        self.group_size = group_size
        self.logger = logger

    def prepare_rollout_batch(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep batch-level workflow hooks visible through group expansion."""
        return self.workflow.prepare_rollout_batch(data)

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, Any] | None:
        from areal.experimental.openai import InteractionWithTokenLogpReward

        results = await asyncio.gather(
            *[self.workflow.arun_episode(engine, data) for _ in range(self.group_size)]
        )

        valid_results = [r for r in results if r is not None]

        # All results None -> return None
        if not valid_results:
            return None

        # Some results None -> warn and continue with valid ones
        if len(valid_results) < len(results):
            self.logger.warning(
                f"GroupedRolloutWorkflow: {len(results) - len(valid_results)}/{len(results)} "
                "trajectories returned None, using remaining results"
            )

        # Check if results are InteractionWithTokenLogpReward dicts
        first = valid_results[0]
        if (
            isinstance(first, dict)
            and first
            and all(
                isinstance(v, InteractionWithTokenLogpReward) for v in first.values()
            )
        ):
            # Merge dicts - each result is {completion_id: InteractionWithTokenLogpReward}
            merged: dict[str, InteractionWithTokenLogpReward] = {}
            for result in valid_results:
                merged.update(result)
            return merged if merged else None

        # Otherwise, tensor dicts - concatenate
        concatenated = concat_padded_tensors(valid_results)
        return concatenated if concatenated else None


class RemoteInfBackendProtocol(Protocol):
    """Protocol defining backend-specific operations for remote inference engines.

    This protocol abstracts the differences between various remote inference servers
    (SGLang, vLLM, etc.) by defining a common interface for:
    - Building HTTP requests with backend-specific formats
    - Parsing backend-specific responses
    - Handling weight updates
    - Managing control flow (pause/resume)
    - Supporting optional features (LoRA)

    Implementations can raise NotImplementedError for unsupported features.
    """

    def build_generation_request(
        self, req: ModelRequest, with_lora: bool, version: int
    ) -> HttpRequest:
        """Build HTTP request for text generation.

        Parameters
        ----------
        req : ModelRequest
            The generation request containing input and parameters
        with_lora : bool
            Whether to specify a LoRA to use
        version : int
            The current weight version for versioned LoRA names

        Returns
        -------
        HttpRequest
            The HTTP request with endpoint and payload
        """
        ...

    def parse_generation_response(
        self, response: dict[str, Any]
    ) -> HttpGenerationResult:
        """Parse generation response into standard format.

        Parameters
        ----------
        response : Dict[str, Any]
            The raw JSON response from the server

        Returns
        -------
        HttpGenerationResult
            Parsed result with tokens, logprobs, and stop reason
        """
        ...

    def build_disk_weight_update_requests(
        self, meta: WeightUpdateMeta
    ) -> WeightUpdateRequests:
        """Build requests for loading weights from disk.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing path and configuration

        Returns
        -------
        WeightUpdateRequests
            Collection of HTTP requests (may be multiple for LoRA workflows)
        """
        ...

    def build_distributed_weight_update_requests(
        self,
        meta: WeightUpdateMeta,
        param_specs: list[ParamSpec],
    ) -> WeightUpdateRequests:
        """Build requests for distributed weight update via NCCL/XCCL.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing communication group info
        param_specs : List[ParamSpec]
            Specifications for parameters to be updated

        Returns
        -------
        WeightUpdateRequests
            Collection of HTTP requests for distributed update
        """
        ...

    def build_init_weights_group_request(
        self, addr: str, server_idx: int, meta: WeightUpdateMeta
    ) -> HttpRequest:
        """Build request to initialize weight update xccl group.

        Parameters
        ----------
        addr : str
            Server address
        server_idx : int
            Index of this server in the server list
        meta : WeightUpdateMeta
            Metadata containing communication backend configuration

        Returns
        -------
        HttpRequest
            The HTTP request to initialize the group
        """
        ...

    def get_pause_request(self) -> HttpRequest:
        """Get request to pause generation.

        Returns
        -------
        HttpRequest
            The HTTP request to pause generation

        Raises
        ------
        NotImplementedError
            If pause is not supported by this backend
        """
        ...

    def get_resume_request(self) -> HttpRequest:
        """Get request to resume generation.

        Returns
        -------
        HttpRequest
            The HTTP request to resume generation

        Raises
        ------
        NotImplementedError
            If resume is not supported by this backend
        """
        ...

    def build_lora_unload_request(
        self, meta: WeightUpdateMeta, version: int
    ) -> HttpRequest:
        """Build request to unload a versioned LoRA adapter."""
        ...

    def get_health_check_request(self) -> HttpRequest:
        """Get the health check request.

        Returns
        -------
        HttpRequest
            The HTTP request for health checks
        """
        ...

    def get_offload_request(self) -> HttpRequest:
        """Get request to offload model memory.

        Returns
        -------
        HttpRequest
            The HTTP request to offload model memory

        Raises
        ------
        NotImplementedError
            If offload is not supported by this backend
        """
        ...

    def get_onload_request(self, tags: list[str] | None = None) -> HttpRequest:
        """Get request to onload model memory.

        Parameters
        ----------
        tags : list[str], optional
            Tags to onload specific components. If None, onloads all components.

        Returns
        -------
        HttpRequest
            The HTTP request to onload model memory

        Raises
        ------
        NotImplementedError
            If onload is not supported by this backend
        """
        ...

    def launch_server(self, server_args: dict[str, Any]) -> subprocess.Popen:
        """Launch inference server subprocess.

        Parameters
        ----------
        server_args : dict[str, Any]
            Server configuration arguments for build_cmd_from_args

        Returns
        -------
        subprocess.Popen
            The launched server process
        """
        ...


class RemoteInfEngine(InferenceEngine):
    """
    Base implementation for HTTP-based remote inference engines.

    This class provides common functionality for communicating with remote
    inference servers via HTTP REST APIs. Backend-specific behaviors are
    delegated to an injected RemoteInfBackendProtocol implementation.

    Uses composition pattern - instantiate directly with a backend rather
    than inheriting from this class.

    Parameters
    ----------
    config : InferenceEngineConfig
        Configuration for the inference engine
    backend : RemoteInfBackendProtocol
        Backend implementation providing server-specific behavior
    """

    def __init__(
        self, config: InferenceEngineConfig, backend: RemoteInfBackendProtocol
    ):
        self.config = config
        self.backend = backend

        self.rid_to_address = {}
        # Maintain the addresses for the recent 128 requests
        self.rid_queue = []
        self.addresses = []
        self.server_idx = 0

        self._version = 0

        self.lock = Lock()
        self._generation_pause_cond = Condition()
        self._generation_paused = False
        self._lora_cond = Condition()
        self._active_lora_versions: Counter[int] = Counter()
        self._loaded_lora_versions_by_addr: dict[str, set[int]] = {}
        self._unloading_lora_versions_by_addr: dict[str, set[int]] = {}
        self._max_loaded_loras_by_addr: dict[str, int] = {}
        self._warned_unknown_lora_capacity = False
        # Weight updates may be submitted from different controller threads. Only
        # one LoRA version may change server adapter state at a time.
        self._lora_update_lock = Lock()

        self._workflow_executor: WorkflowExecutor | None = None
        # ``RolloutController.prepare_batch`` streams individual tasks to remote
        # workers, so its batch-level workflow hook is coordinated explicitly on
        # one worker before those tasks are fanned out. Keep that hook's workflow
        # instance alive across dataloader batches: hooks such as smooth weighted
        # stratification carry residual state between calls.
        self._batch_preparation_workflow: RolloutWorkflow | None = None
        self._batch_preparation_lock = Lock()
        self._initialized = False
        self._proxy_gateway_addr: str | None = None
        self.local_server_processes: list[LocalInfServerInfo] = []

    def _wait_for_server(self, address: str, process: subprocess.Popen | None = None):
        """Wait for a server to become healthy."""
        try:
            host, port = split_hostport(address)
            base_url = f"http://{format_hostport(host, port)}"
        except ValueError:
            base_url = f"http://{address}"
        tik = time.time()
        while time.time() - tik < self.config.setup_timeout:
            if self.check_health(base_url):
                return
            time.sleep(1)
        raise TimeoutError("server launch failed")

    def check_health(self, base_url):
        """Check if server is healthy."""
        try:
            health_req = self.backend.get_health_check_request()
            url = f"{base_url}{health_req.endpoint}"
            response = requests.request(
                health_req.method, url, json=health_req.payload, timeout=30
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    @staticmethod
    def _parse_versioned_lora_name(lora_name: str) -> int | None:
        match = re.search(r"-v(\d+)$", lora_name)
        return int(match.group(1)) if match is not None else None

    def _record_lora_server_args(
        self, server_args: dict[str, Any], addr: str | None = None
    ) -> None:
        max_loaded_loras = server_args.get(
            "max_loaded_loras", server_args.get("max_loras")
        )
        max_loaded_loras = (
            int(max_loaded_loras) if max_loaded_loras is not None else None
        )

        lora_paths = (
            server_args.get("lora_paths") or server_args.get("lora_modules") or []
        )
        loaded_versions: set[int] = set()
        for lora_spec in lora_paths:
            lora_name = str(lora_spec).split("=", 1)[0]
            version = self._parse_versioned_lora_name(lora_name)
            if version is not None:
                loaded_versions.add(version)

        addrs = [addr] if addr is not None else list(self.addresses)
        if not addrs:
            return

        with self._lora_cond:
            for server_addr in addrs:
                self._loaded_lora_versions_by_addr.setdefault(
                    server_addr, set()
                ).update(loaded_versions)
                self._unloading_lora_versions_by_addr.setdefault(server_addr, set())
                if max_loaded_loras is not None:
                    self._max_loaded_loras_by_addr[server_addr] = max_loaded_loras
            self._lora_cond.notify_all()

    def record_lora_server_args(
        self, server_args: dict[str, Any], addresses: list[str] | None = None
    ) -> None:
        """Record LoRA capacity and initial adapters for known server addresses."""
        if addresses is None:
            self._record_lora_server_args(server_args)
            return
        for addr in addresses:
            self._record_lora_server_args(server_args, addr)

    def _log_lora_capacity_warning_once(self) -> None:
        if self._warned_unknown_lora_capacity:
            return
        self._warned_unknown_lora_capacity = True
        self.logger.warning(
            "LoRA capacity is unknown for one or more inference servers; "
            "AReaL will load new LoRA adapters without proactive unload on those servers."
        )

    @contextmanager
    def lora_version_lease(self, version: int | None, *, workflow: bool = False):
        """Keep a LoRA version loaded while a request or workflow may use it.

        Passing ``None`` atomically leases the current version. This prevents an
        updater from choosing that version for unload between a separate
        ``get_version()`` call and lease registration.

        ``workflow`` documents the caller scope. Unload safety is per LoRA
        version, not global across all active workflows.
        """
        if not self.config.use_lora:
            yield self.get_version() if version is None else int(version)
            return

        with self._lora_cond:
            if version is None:
                with self.lock:
                    leased_version = self._version
            else:
                leased_version = int(version)

            unloading_addrs = [
                addr
                for addr, versions in self._unloading_lora_versions_by_addr.items()
                if leased_version in versions
            ]
            known_addrs = [
                addr
                for addr in self.addresses
                if addr in self._loaded_lora_versions_by_addr
            ]
            missing_addrs = [
                addr
                for addr in known_addrs
                if leased_version
                not in self._loaded_lora_versions_by_addr.get(addr, set())
            ]
            if unloading_addrs or missing_addrs:
                raise RuntimeError(
                    f"LoRA version {leased_version} is not available for a new lease. "
                    f"unloading_addrs={unloading_addrs}, missing_addrs={missing_addrs}."
                )
            self._active_lora_versions[leased_version] += 1
            self._lora_cond.notify_all()
        try:
            yield leased_version
        finally:
            with self._lora_cond:
                self._active_lora_versions[leased_version] -= 1
                if self._active_lora_versions[leased_version] <= 0:
                    del self._active_lora_versions[leased_version]
                self._lora_cond.notify_all()

    def _set_generation_paused(self, paused: bool) -> None:
        with self._generation_pause_cond:
            self._generation_paused = paused
            self._generation_pause_cond.notify_all()

    def is_generation_paused(self) -> bool:
        with self._generation_pause_cond:
            return self._generation_paused

    def initialize(
        self,
        engine_id: str | None = None,
        addr: str | list[str] | None = None,
        train_data_parallel_size: int | None = None,
    ):
        """Initialize the engine by discovering and connecting to servers.

        Parameters
        ----------
        engine_id : Optional[str]
            Unique identifier for this engine instance
        addr : str | List[str] | None
            Server address(es) to connect to. If None, will auto-discover.
        train_data_parallel_size : int | None
            Data parallel size of the training engine
        """
        if engine_id is None:
            if dist.is_initialized():
                engine_id = str(dist.get_rank())
            else:
                engine_id = uuid.uuid4().hex
        self.engine_id = engine_id
        self.logger = logging.getLogger(f"[RemoteInfEngine Rank {engine_id}]")

        if addr:
            self.addresses = addr if isinstance(addr, list) else [addr]
            self.logger.info("Get server addresses from the `addr` argument.")
        elif len(self.local_server_processes) > 0:
            self.addresses = [
                format_hostport(s.host, s.port) for s in self.local_server_processes
            ]
            self.logger.info("Get server addresses from the local subprocess.")
        elif (
            self.config.experiment_name is not None
            and self.config.trial_name is not None
        ):
            try:
                self.addresses = wait_llm_server_addrs(
                    experiment_name=self.config.experiment_name,
                    trial_name=self.config.trial_name,
                    timeout=1,
                )
                self.logger.info("Get server addresses from name_resolve.")
            except (TimeoutError, RuntimeError):
                self.logger.info(
                    "Failed to get server addresses from name_resolve, "
                    "falling back to environment variable."
                )
                addrs_str = os.getenv("AREAL_LLM_SERVER_ADDRS")
                if addrs_str:
                    # When addr is not provided, fallback to reading addrs from env var
                    self.addresses = addrs_str.split(",")
                    self.logger.info("Get server addresses from environment variable.")

        if not self.addresses:
            raise RuntimeError(
                "No configured inference servers. "
                "Please pass in server addresses by arguments "
                "for `initialize` or environment "
                "variable `AREAL_LLM_SERVER_ADDRS`."
            )

        self.logger.info("Waiting for server ready...")
        for addr_ in self.addresses:
            self._wait_for_server(addr_)
        self.server_idx = random.randint(0, len(self.addresses) - 1)
        self.logger.info("Servers are all ready!")

        self.workflow_executor = WorkflowExecutor(
            config=self.config,
            inference_engine=self,
        )
        self.workflow_executor.initialize(
            logger=self.logger, train_data_parallel_size=train_data_parallel_size
        )
        self._initialized = True

    def destroy(self):
        """Destroy the engine and clean up resources."""
        self._initialized = False
        if self._workflow_executor is not None:
            self._workflow_executor.destroy()
        if len(self.local_server_processes) > 0:
            self.teardown_server()

    @property
    def workflow_executor(self) -> WorkflowExecutor:
        """Get the workflow executor of the inference engine."""
        if self._workflow_executor is None:
            raise RuntimeError("WorkflowExecutor is not initialized")
        return self._workflow_executor

    @workflow_executor.setter
    def workflow_executor(self, workflow_executor: WorkflowExecutor):
        """Set the workflow executor of the inference engine."""
        self._workflow_executor = workflow_executor
        min_allowed = self.get_version() - self.config.max_head_offpolicyness
        workflow_executor.set_min_allowed_rollout_version(min_allowed)

    @property
    def initialized(self) -> bool:
        return self._initialized

    def set_version(self, version):
        """Set the current weight version."""
        version = int(version)
        with self.lock:
            self._version = version
        if self._workflow_executor is not None:
            min_allowed = version - self.config.max_head_offpolicyness
            self._workflow_executor.set_min_allowed_rollout_version(min_allowed)

    def get_version(self):
        """Get the current weight version."""
        with self.lock:
            return self._version

    def _raise_if_rollout_stale(self) -> None:
        ctx = workflow_context.get()
        lora_version = ctx.lora_version
        if lora_version is None or self._workflow_executor is None:
            return
        if self._workflow_executor.is_rollout_version_stale(lora_version):
            raise RolloutStaleError(
                f"Rollout task {ctx.task_id} version {lora_version} is stale "
                f"at current version {self.get_version()}."
            )

    def set_proxy_gateway_addr(self, addr: str) -> None:
        """Set the proxy gateway address.

        Called by ``RolloutController.start_proxy_gateway()`` via
        collective RPC after the proxy gateway is started.
        """
        # HACK: We'd better not have this kind of setter beyond well-defined APIs,
        # but for now it's a workaround for the next release.
        self._proxy_gateway_addr = addr

    def _wrap_openai_agent(self, agent: Any, proxy_addr: str) -> RolloutWorkflow:
        """Wrap an agent workflow in OpenAIProxyWorkflow (HTTP mode only).

        Parameters
        ----------
        agent : Any | None
            The agent workflow to wrap (any class with async run() method).
            ``None`` is valid when ``mode='online'``.
        proxy_addr : str
            HTTP address of the proxy server (required)
        """
        from areal.experimental.openai import OpenAIProxyWorkflow

        openai_cfg = self.config.openai or OpenAIProxyConfig()

        return OpenAIProxyWorkflow(
            mode=openai_cfg.mode,
            agent=agent,
            proxy_addr=proxy_addr,
            admin_api_key=openai_cfg.admin_api_key,
            discount=openai_cfg.turn_discount,
            export_style=openai_cfg.export_style,
            subproc_max_workers=openai_cfg.subproc_max_workers,
            proxy_gateway_addr=self._proxy_gateway_addr,
        )

    def _resolve_workflow(
        self,
        workflow: WorkflowLike | None,
        workflow_kwargs: dict[str, Any] | None,
        group_size: int = 1,
        proxy_addr: str | None = None,
    ) -> RolloutWorkflow:
        resolved: RolloutWorkflow

        # 0. None workflow = online mode (config-driven)
        if workflow is None:
            openai_cfg = self.config.openai or OpenAIProxyConfig()
            if openai_cfg.mode != "online":
                raise ValueError(
                    "workflow is None but OpenAIProxyConfig.mode is not 'online'. "
                    "Provide a workflow or set mode='online' in the config."
                )
            if proxy_addr is None:
                raise ValueError("proxy_addr is required for online mode")
            resolved = self._wrap_openai_agent(None, proxy_addr=proxy_addr)
            if group_size > 1:
                resolved = GroupedRolloutWorkflow(resolved, group_size, self.logger)
            return resolved

        # 1. Already a RolloutWorkflow instance
        if isinstance(workflow, RolloutWorkflow):
            if workflow_kwargs is not None:
                self.logger.warning(
                    "workflow_kwargs is ignored when workflow is already an instance"
                )
            resolved = workflow

        # 2. RolloutWorkflow class
        elif isinstance(workflow, type) and issubclass(workflow, RolloutWorkflow):
            if workflow_kwargs is None:
                raise ValueError(
                    f"workflow_kwargs is required when workflow is a class. "
                    f"Got workflow={workflow}, but workflow_kwargs=None."
                )
            resolved = workflow(**workflow_kwargs)

        # 3. String import path
        elif isinstance(workflow, str):
            try:
                imported_obj = import_from_string(workflow)
            except (ValueError, ImportError, AttributeError) as e:
                raise ValueError(
                    f"Failed to import workflow from string {workflow!r}: {e}"
                ) from e

            # Check if it's a RolloutWorkflow class
            if isinstance(imported_obj, type) and issubclass(
                imported_obj, RolloutWorkflow
            ):
                if workflow_kwargs is None:
                    raise ValueError(
                        f"workflow_kwargs is required when workflow is a class or string. "
                        f"Got workflow={workflow}, but workflow_kwargs=None."
                    )
                resolved = imported_obj(**workflow_kwargs)

            # Check if it's a RolloutWorkflow instance
            elif isinstance(imported_obj, RolloutWorkflow):
                if workflow_kwargs is not None:
                    self.logger.warning(
                        "workflow_kwargs is ignored when workflow resolves to an instance"
                    )
                resolved = imported_obj

            # Otherwise, treat it as an agent-like workflow (needs proxy)
            else:
                if proxy_addr is None:
                    raise ValueError(
                        f"proxy_addr is required for agent workflows (non-RolloutWorkflow). "
                        f"Ensure proxy workers are initialized via RolloutController.start_proxy(). "
                        f"Got workflow={workflow!r}"
                    )

                # Instantiate if it's a class
                if isinstance(imported_obj, type):
                    agent = imported_obj(**(workflow_kwargs or {}))
                else:
                    # Already an instance
                    agent = imported_obj

                resolved = self._wrap_openai_agent(agent, proxy_addr=proxy_addr)

        # 4. Callable class (agent-like workflow)
        elif isinstance(workflow, type):
            if proxy_addr is None:
                raise ValueError(
                    "proxy_addr is required for agent workflows (non-RolloutWorkflow). "
                    "Ensure proxy workers are initialized via RolloutController.start_proxy()."
                )
            agent = workflow(**(workflow_kwargs or {}))
            resolved = self._wrap_openai_agent(agent, proxy_addr=proxy_addr)

        # 5. Instance of agent-like workflow
        else:
            if proxy_addr is None:
                raise ValueError(
                    "proxy_addr is required for agent workflows (non-RolloutWorkflow). "
                    "Ensure proxy workers are initialized via RolloutController.start_proxy()."
                )
            if workflow_kwargs is not None:
                self.logger.warning(
                    "workflow_kwargs is ignored when workflow is already an instance"
                )
            resolved = self._wrap_openai_agent(workflow, proxy_addr=proxy_addr)

        # Wrap with GroupedRolloutWorkflow if group_size > 1
        if group_size > 1:
            resolved = GroupedRolloutWorkflow(resolved, group_size, self.logger)

        return resolved

    def prepare_rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow: WorkflowLike | None,
        workflow_kwargs: dict[str, Any] | None = None,
        group_size: int = 1,
        proxy_addr: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run the stateful batch hook for controller-driven rollouts.

        The controller captures one workflow on its first ``prepare_batch`` call,
        matching :class:`WorkflowExecutor`. This companion instance exists only
        to coordinate batch-level input decisions; the prepared dictionaries are
        subsequently sent to the normal per-task workflow instances.
        """
        with self._batch_preparation_lock:
            if self._batch_preparation_workflow is None:
                self._batch_preparation_workflow = self._resolve_workflow(
                    workflow,
                    workflow_kwargs,
                    group_size,
                    proxy_addr=proxy_addr,
                )
            return WorkflowExecutor._prepare_rollout_batch(
                data, self._batch_preparation_workflow
            )

    def _resolve_should_accept_fn(
        self, should_accept_fn: Callable[[dict[str, Any]], bool] | str | None
    ) -> Callable[[dict[str, Any]], bool] | None:
        """Resolve should_accept_fn parameter to a callable or None.

        Parameters
        ----------
        should_accept_fn : Callable[[Dict[str, Any]], bool] | str | None
            The should_accept_fn specification

        Returns
        -------
        Callable[[Dict[str, Any]], bool] | None
            A callable for trajectory filtering, or None

        Raises
        ------
        ValueError
            If string import fails
        TypeError
            If imported object is not callable
        """
        if should_accept_fn is None or callable(should_accept_fn):
            return should_accept_fn

        if isinstance(should_accept_fn, str):
            try:
                func = import_from_string(should_accept_fn)
            except (ValueError, ImportError, AttributeError) as e:
                raise ValueError(
                    f"Failed to import should_accept_fn from string {should_accept_fn!r}: {e}"
                ) from e
            if not callable(func):
                raise TypeError(
                    f"Imported object {func} from {should_accept_fn!r} is not callable"
                )
            return func

        raise TypeError(
            f"Invalid should_accept_fn type: {type(should_accept_fn)}. "
            f"Expected callable or string module path."
        )

    def choose_server(self) -> str:
        """Choose a server based on the scheduling policy.

        Returns
        -------
        str
            Selected server address

        Raises
        ------
        NotImplementedError
            If schedule policy other than round-robin is used
        """
        if self.config.schedule_policy == "round_robin":
            server = self.addresses[self.server_idx]
            self.server_idx = (self.server_idx + 1) % len(self.addresses)
            return server
        raise NotImplementedError("Only round-robin scheduling is implemented.")

    async def agenerate(self, req: ModelRequest) -> ModelResponse:
        """Asynchronously generate a response for the given request.

        Parameters
        ----------
        req : ModelRequest
            The model request containing input data and generation parameters

        Returns
        -------
        ModelResponse
            The generated response from the model
        """
        # Create a shallow copy of the input request
        # we are going to modify it in-place
        req = req.copy()

        request_max_attempts = req.metadata.pop(
            _REQUEST_MAX_ATTEMPTS_METADATA_KEY, None
        )
        if request_max_attempts is None:
            request_max_attempts = self.config.request_retries
        elif (
            isinstance(request_max_attempts, bool)
            or not isinstance(request_max_attempts, int)
            or request_max_attempts < 1
        ):
            raise ValueError(
                f"ModelRequest.metadata[{_REQUEST_MAX_ATTEMPTS_METADATA_KEY!r}] "
                "must be a positive integer."
            )

        # Populate return_routed_experts from config to metadata
        if self.config.return_routed_experts:
            req.metadata["return_routed_experts"] = True

        # Validate n_samples
        gconfig = req.gconfig
        if gconfig.n_samples != 1:
            raise ValueError(
                "Inference engines do not support n_samples > 1. "
                "Please call generate multiple times with n_samples = 1."
            )

        # Validate max_new_tokens
        max_new_tokens = min(
            gconfig.max_tokens - len(req.input_ids), gconfig.max_new_tokens
        )
        if max_new_tokens <= 0:
            raise RuntimeError(
                f"max_new_tokens ({max_new_tokens}) is non-positive! "
                f"max_tokens={gconfig.max_tokens}, prompt_len={len(req.input_ids)}, "
                f"max_new_tokens={gconfig.max_new_tokens}."
            )

        # Update max_new_tokens in request
        req.gconfig.max_new_tokens = max_new_tokens

        # Make request
        start_time = time.perf_counter()
        accumulated_output_tokens = []
        accumulated_output_logprobs = []
        accumulated_versions = []
        accumulated_routed_experts: list[np.ndarray] = []
        request_lora_version = None
        request_disable_lora = bool(req.metadata.get("disable_lora", False))
        if self.config.use_lora and "lora_version" in req.metadata:
            try:
                request_lora_version = int(req.metadata["lora_version"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "ModelRequest.metadata['lora_version'] must be an integer."
                ) from exc
            if request_lora_version < 0:
                raise ValueError(
                    "ModelRequest.metadata['lora_version'] must be non-negative."
                )
        context_lora_enabled = self.config.use_lora and not request_disable_lora
        if request_lora_version is None and context_lora_enabled:
            context_lora_version = workflow_context.get().lora_version
            if context_lora_version is not None:
                request_lora_version = int(context_lora_version)

        # A single "rid" shares the same server to allow KV cache reuse
        if req.rid in self.rid_to_address:
            server_addr = self.rid_to_address[req.rid]
        else:
            server_addr = self.choose_server()
            if len(self.rid_queue) >= RID_CACHE_SIZE:
                # Remove the oldest entry if cache is full
                oldest_rid = self.rid_queue.pop(0)
                self.rid_to_address.pop(oldest_rid, None)
            self.rid_to_address[req.rid] = server_addr
            self.rid_queue.append(req.rid)

        # Get the shared session from workflow context
        session = await workflow_context.get_aiohttp_session()

        # Deal with generation interruption
        stop_reason = None
        ori_max_new_tokens = gconfig.max_new_tokens
        while (
            stop_reason not in ["stop", "tool_calls", "length"]
            and len(accumulated_output_tokens) < ori_max_new_tokens
        ):
            # Only hard generation pause is honored here. Rollout submission
            # pause stops new workflows, but in-flight workflows may continue.
            while self.is_generation_paused():
                await asyncio.sleep(0.5)

            self._raise_if_rollout_stale()
            with_lora = self.config.use_lora and not request_disable_lora
            lora_lease = (
                self.lora_version_lease(request_lora_version)
                if with_lora
                else nullcontext()
            )
            with lora_lease as leased_version:
                generation_version = leased_version if with_lora else self.get_version()
                http_req = self.backend.build_generation_request(
                    req,
                    with_lora=with_lora,
                    version=generation_version,
                )
                result = await arequest_with_retry(
                    session=session,
                    addr=server_addr,
                    endpoint=http_req.endpoint,
                    payload=http_req.payload,
                    method=http_req.method,
                    max_retries=request_max_attempts,
                    timeout=self.config.request_timeout,
                )
            self._raise_if_rollout_stale()

            # Assert response is JSON dict (not text/binary from error pages)
            if not isinstance(result, dict):
                raise ValueError(
                    f"Expected JSON dict response, got {type(result).__name__}"
                )

            # Parse response using backend
            gen_result = self.backend.parse_generation_response(result)
            stop_reason = gen_result.stop_reason

            if (
                req.metadata.get("return_routed_experts", False)
                and gen_result.routed_experts is None
            ):
                if stop_reason != "abort":  # Only validate for successful generations
                    raise RuntimeError(
                        "Requested return_routed_experts=True but received None from SGLang. "
                        "This usually means the model is not a MoE (Mixture of Experts) model. "
                        "Please use a MoE model to get routed_experts information."
                    )

            # Update accumulated outputs
            accumulated_output_tokens.extend(gen_result.output_tokens)
            accumulated_output_logprobs.extend(gen_result.output_logprobs)
            accumulated_versions.extend(
                [generation_version] * len(gen_result.output_tokens)
            )
            # Accumulate routed_experts for MoE models
            if gen_result.routed_experts is not None:
                accumulated_routed_experts.append(gen_result.routed_experts)

            # Update request for next iteration
            req.input_ids += gen_result.output_tokens
            req.gconfig.max_new_tokens -= len(gen_result.output_tokens)
            assert req.gconfig.max_new_tokens >= 0, (
                req.gconfig.max_new_tokens,
                len(gen_result.output_tokens),
                len(req.input_ids),
            )

        # Final abort handling
        if stop_reason == "abort":
            # If stop_reason is "abort", the only reason we exit the loop is
            # len(accumulated_output_tokens) >= gconfig.max_new_tokens
            # so the actual reason is length
            stop_reason = "length"

        latency = time.perf_counter() - start_time

        accumulated_routed_experts = (
            np.concatenate(accumulated_routed_experts)
            if accumulated_routed_experts
            else None
        )

        response = ModelResponse(
            input_tokens=req.input_ids[
                : len(req.input_ids) - len(accumulated_output_tokens)
            ],
            input_images=req.image_data,
            output_tokens=accumulated_output_tokens,
            output_logprobs=accumulated_output_logprobs,
            output_versions=accumulated_versions,
            stop_reason=stop_reason,
            latency=latency,
            ttft=latency,  # Simplified for non-streaming
            tokenizer=req.tokenizer,
            processor=req.processor,
            routed_experts=accumulated_routed_experts,
        )
        return response

    def init_weights_update_group(
        self, meta: WeightUpdateMeta, xccl_group_ranks: list[int] | None = None
    ) -> Future[None]:
        """Initialize the weight update process group for distributed weight updates.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing information about the weight update
        xccl_group_ranks : list[int] | None, optional
            Explicit rank assignment for each remote inference worker, aligned with
            ``self.addresses`` (same length, same order).

            - If provided, worker at ``self.addresses[i]`` will initialize the
            communication group using rank ``xccl_group_ranks[i]``.
            - If None, ranks are assigned by address order: rank ``i`` for
            ``self.addresses[i]``.

        Returns
        -------
        Future[None]
            A future object representing the asynchronous initialization operation
        """
        assert meta.type == "xccl"

        fut = get_executor().submit(
            _init_weights_update_group_remote,
            self.backend,
            meta,
            self.addresses,
            self.config.request_timeout,
            xccl_group_ranks,
        )

        def callback(fut):
            if fut.cancelled():
                return
            if fut.exception() is not None:
                self.logger.error(
                    "Failed to initialize %s group for distributed weight update for %s: %s",
                    current_platform.communication_backend.upper(),
                    meta.nccl_group_name,
                    repr(fut.exception()),
                )
                return
            self.logger.info(
                f"Initialized {current_platform.communication_backend.upper()} group "
                f"for distributed weight update for {meta.nccl_group_name}."
            )

        fut.add_done_callback(callback)
        return fut

    def update_weights_from_distributed(
        self, meta: WeightUpdateMeta, param_specs: list[ParamSpec]
    ) -> Future[None]:
        """Update weights in the inference engine from distributed memory.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing information about the weight update
        param_specs : List[ParamSpec]
            A list of parameter specifications for the weights to be updated

        Returns
        -------
        Future[None]
            A future object representing the asynchronous weight update operation
        """
        assert meta.type == "xccl"

        fut = get_executor().submit(
            _update_weights_from_distributed,
            self.backend,
            meta,
            param_specs,
            self.addresses,
            self.config.request_timeout,
        )

        return fut

    def update_weights_from_disk(self, meta: WeightUpdateMeta) -> Future[None]:
        """Update weights in the inference engine from disk.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing information about the weight update

        Returns
        -------
        Future[None]
            A future object representing the asynchronous weight update operation
        """
        assert meta.type == "disk"

        tik = time.perf_counter()

        # Use ProcessPool to bypass python GIL for running async coroutines
        if self.config.experiment_name is None or self.config.trial_name is None:
            raise RuntimeError(
                "Experiment and trial names must be set for disk-based weight updates."
            )

        if meta.use_lora:
            fut = get_executor().submit(
                self._update_lora_weights_from_disk,
                self.config.experiment_name,
                self.config.trial_name,
                self.get_version(),
                meta,
            )
        else:
            fut = get_executor().submit(
                _update_weights_from_disk,
                self.backend,
                self.config.experiment_name,
                self.config.trial_name,
                self.get_version(),
                self.addresses,
                meta,
                self.config.request_retries,
                self.config.request_timeout,
            )

        def callback(fut):
            respond_time = fut.result()
            self.logger.info(
                f"Loading weights from disk done "
                f"in {(time.perf_counter() - tik):.2f}s. "
                f"Respond time: {respond_time:.2f}s."
            )
            if meta.clear_checkpoint_after_load:
                shutil.rmtree(meta.path, ignore_errors=True)

        fut.add_done_callback(callback)
        return fut

    def _select_inactive_loras_to_unload(
        self, new_version: int, deadline: float
    ) -> dict[int, list[str]]:
        while True:
            with self._lora_cond:
                unload_plan: dict[int, list[str]] = defaultdict(list)
                blocked_addrs: list[str] = []
                unknown_capacity_addrs: list[str] = []

                for addr in list(self.addresses):
                    loaded_versions = self._loaded_lora_versions_by_addr.setdefault(
                        addr, set()
                    )
                    unloading_versions = (
                        self._unloading_lora_versions_by_addr.setdefault(addr, set())
                    )

                    if new_version in loaded_versions:
                        continue

                    max_loaded_loras = self._max_loaded_loras_by_addr.get(addr)
                    if max_loaded_loras is None:
                        unknown_capacity_addrs.append(addr)
                        continue

                    if len(loaded_versions) < max_loaded_loras:
                        continue

                    candidates = [
                        version
                        for version in sorted(loaded_versions)
                        if version not in self._active_lora_versions
                        and version not in unloading_versions
                    ]
                    if candidates:
                        unload_plan[candidates[0]].append(addr)
                    else:
                        blocked_addrs.append(addr)

                if unknown_capacity_addrs:
                    self._log_lora_capacity_warning_once()

                if not blocked_addrs:
                    for version, addrs in unload_plan.items():
                        for addr in addrs:
                            self._unloading_lora_versions_by_addr.setdefault(
                                addr, set()
                            ).add(version)
                    return dict(unload_plan)

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    active_versions = dict(self._active_lora_versions)
                    loaded_by_addr = {
                        addr: sorted(versions)
                        for addr, versions in self._loaded_lora_versions_by_addr.items()
                    }
                    unloading_by_addr = {
                        addr: sorted(versions)
                        for addr, versions in self._unloading_lora_versions_by_addr.items()
                    }
                    raise TimeoutError(
                        "Timed out waiting for inactive LoRA adapter slots. "
                        f"new_version={new_version}, "
                        f"blocked_addrs={blocked_addrs}, "
                        f"loaded_versions_by_addr={loaded_by_addr}, "
                        f"active_versions={active_versions}, "
                        f"unloading_versions_by_addr={unloading_by_addr}, "
                        f"max_loaded_loras_by_addr={self._max_loaded_loras_by_addr}."
                    )
                self._lora_cond.wait(timeout=min(1.0, remaining))

    def _run_requests_on_servers(
        self, requests: list[HttpRequest], addresses: list[str]
    ) -> tuple[set[str], dict[str, BaseException]]:
        addresses = list(addresses)
        if not addresses:
            return set(), {}

        async def _fn():
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.request_timeout),
                read_bufsize=1024 * 1024 * 10,
                connector=get_default_connector(),
            ) as session:

                async def _run_for_addr(addr: str):
                    for http_req in requests:
                        await arequest_with_retry(
                            session=session,
                            addr=addr,
                            endpoint=http_req.endpoint,
                            payload=http_req.payload,
                            method=http_req.method,
                            max_retries=self.config.request_retries,
                            timeout=self.config.request_timeout,
                        )

                results = await asyncio.gather(
                    *[_run_for_addr(addr) for addr in addresses],
                    return_exceptions=True,
                )
                successes: set[str] = set()
                failures: dict[str, BaseException] = {}
                for addr, result in zip(addresses, results, strict=True):
                    if isinstance(result, BaseException):
                        failures[addr] = result
                    else:
                        successes.add(addr)
                return successes, failures

        return uvloop.run(_fn())

    def _run_requests_on_all_servers(self, requests: list[HttpRequest]) -> None:
        async def _fn():
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.request_timeout),
                read_bufsize=1024 * 1024 * 10,
                connector=get_default_connector(),
            ) as session:
                for http_req in requests:
                    jobs = [
                        arequest_with_retry(
                            session=session,
                            addr=addr,
                            endpoint=http_req.endpoint,
                            payload=http_req.payload,
                            method=http_req.method,
                            max_retries=self.config.request_retries,
                            timeout=self.config.request_timeout,
                        )
                        for addr in self.addresses
                    ]
                    await asyncio.gather(*jobs)

        uvloop.run(_fn())

    @staticmethod
    def _format_lora_server_failures(
        action: str, version: int, failures: dict[str, BaseException]
    ) -> str:
        details = "; ".join(
            f"{addr}: {type(exc).__name__}: {exc}" for addr, exc in failures.items()
        )
        return (
            f"Failed to {action} LoRA version {version} "
            f"on {len(failures)} server(s): {details}"
        )

    def _unload_lora_versions(
        self, meta: WeightUpdateMeta, unload_plan: dict[int, list[str]]
    ) -> None:
        all_failures: dict[int, dict[str, BaseException]] = {}
        try:
            for version, addrs in unload_plan.items():
                unload_req = self.backend.build_lora_unload_request(meta, version)
                successes, failures = self._run_requests_on_servers([unload_req], addrs)

                with self._lora_cond:
                    for addr in successes:
                        self._loaded_lora_versions_by_addr.setdefault(
                            addr, set()
                        ).discard(version)
                        self._unloading_lora_versions_by_addr.setdefault(
                            addr, set()
                        ).discard(version)
                    for addr in failures:
                        self._unloading_lora_versions_by_addr.setdefault(
                            addr, set()
                        ).discard(version)
                    self._lora_cond.notify_all()

                if failures:
                    all_failures[version] = failures
        except Exception:
            with self._lora_cond:
                for version, addrs in unload_plan.items():
                    for addr in addrs:
                        self._unloading_lora_versions_by_addr.setdefault(
                            addr, set()
                        ).discard(version)
                self._lora_cond.notify_all()
            raise

        if all_failures:
            messages = [
                self._format_lora_server_failures("unload", version, failures)
                for version, failures in all_failures.items()
            ]
            raise RuntimeError("; ".join(messages))

    def _load_lora_version_from_disk(
        self, meta: WeightUpdateMeta, requests: list[HttpRequest], addrs: list[str]
    ) -> None:
        assert meta.version is not None
        successes, failures = self._run_requests_on_servers(requests, addrs)

        with self._lora_cond:
            for addr in successes:
                self._loaded_lora_versions_by_addr.setdefault(addr, set()).add(
                    meta.version
                )
            self._lora_cond.notify_all()

        if failures:
            raise RuntimeError(
                self._format_lora_server_failures("load", meta.version, failures)
            )

    def _update_lora_weights_from_disk(
        self,
        experiment_name: str,
        trial_name: str,
        model_version: int,
        meta: WeightUpdateMeta,
    ):
        if not self._lora_update_lock.acquire(blocking=False):
            raise RuntimeError(
                "A LoRA weight update is already running; concurrent updates are "
                "not supported."
            )
        try:
            return self._update_lora_weights_from_disk_serial(
                experiment_name,
                trial_name,
                model_version,
                meta,
            )
        finally:
            self._lora_update_lock.release()

    def _update_lora_weights_from_disk_serial(
        self,
        experiment_name: str,
        trial_name: str,
        model_version: int,
        meta: WeightUpdateMeta,
    ):
        if meta.version is None:
            raise ValueError("Version is required for LoRA update.")

        update_name = names.update_weights_from_disk(
            experiment_name, trial_name, model_version
        )
        save_timestamp = float(
            name_resolve.wait(update_name, timeout=self.config.request_timeout)
        )
        load_timestamp = datetime.now().timestamp()

        deadline = time.monotonic() + self.config.request_timeout
        unload_plan = self._select_inactive_loras_to_unload(meta.version, deadline)
        if unload_plan:
            self._unload_lora_versions(meta, unload_plan)

        with self._lora_cond:
            load_addrs = [
                addr
                for addr in self.addresses
                if meta.version
                not in self._loaded_lora_versions_by_addr.setdefault(addr, set())
            ]

        if load_addrs:
            weight_reqs = self.backend.build_disk_weight_update_requests(meta)
            self._load_lora_version_from_disk(meta, weight_reqs.requests, load_addrs)

        return load_timestamp - save_timestamp

    def submit(
        self,
        data: dict[str, Any],
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Callable[[dict[str, Any]], bool] | str | None = None,
        group_size: int = 1,
        task_id: int | None = None,
        callback_addr: str | None = None,
        is_eval: bool = False,
        proxy_addr: str | None = None,
    ) -> int:
        """Submit a request to the inference engine and return immediately.

        Parameters
        ----------
        data : Dict[str, Any]
            The input data for rollout
        workflow : WorkflowLike
            The workflow to use for rollout generation
        workflow_kwargs : Dict[str, Any], optional
            Keyword arguments to pass to the workflow constructor
        should_accept_fn : Callable[[Dict[str, Any]], bool] | str, optional
            A function or module path for trajectory filtering
        group_size : int
            Number of times to run the workflow per input and concatenate results.
            Default is 1 (no grouping).
        task_id : int, optional
            The task ID to use. If None, a new task ID will be generated internally.
        is_eval : bool, optional
            Whether this is an evaluation workflow. Affects variables like trajectory dump path
            and statistics keys. By default False.
        proxy_addr : str, optional
            HTTP address of the proxy server for AgentWorkflow. If provided,
            AgentWorkflow will use this proxy instead of a local one.
        """
        if workflow is None and (
            self.config.openai is None or self.config.openai.mode != "online"
        ):
            raise ValueError(
                "workflow must be specified for submit (unless mode='online')"
            )
        if callback_addr:
            self.workflow_executor.dispatcher.register_callback(task_id, callback_addr)

        # Resolve workflow to a RolloutWorkflow instance
        resolved_workflow = self._resolve_workflow(
            workflow, workflow_kwargs, group_size, proxy_addr=proxy_addr
        )
        resolved_should_accept_fn = self._resolve_should_accept_fn(should_accept_fn)

        return self.workflow_executor.submit(
            data,
            workflow=resolved_workflow,
            should_accept_fn=resolved_should_accept_fn,
            task_id=task_id,
            is_eval=is_eval,
        )

    def wait(
        self, count: int, timeout: float | None = None, raise_timeout: bool = True
    ) -> list[dict[str, Any] | None]:
        """Wait for a specified number of requests to complete.

        Parameters
        ----------
        count : int
            The number of accepted trajectories to wait for
        timeout : float, optional
            Timeout in seconds
        raise_timeout : bool, optional
            Whether to raise a TimeoutError when the timeout is exceeded, by default True

        Returns
        -------
        list[dict[str, Any] | None]
            A list of trajectory dictionaries. Each element may be None for rejected trajectories.
            Returns an empty list if timeout exceeded and raise_timeout is False.
        """
        return self.workflow_executor.wait(
            count, timeout=timeout, raise_timeout=raise_timeout
        )

    def wait_for_task(
        self, task_id: int, timeout: float | None = None, raise_timeout: bool = True
    ) -> dict[str, Any] | None:
        """Wait for a specific submitted task to complete."""
        return self.workflow_executor.wait_for_task(task_id, timeout, raise_timeout)

    def wait_for_task_with_metadata(
        self, task_id: int, timeout: float | None = None, raise_timeout: bool = True
    ):
        """Wait for a task while preserving its authoritative rollout version."""
        return self.workflow_executor.wait_for_task_with_metadata(
            task_id, timeout, raise_timeout
        )

    def _discard_task(self, task_id: int, tombstone_ttl_seconds: float = 60.0) -> bool:
        """Internal controller hook to discard a timed-out task result."""
        return self.workflow_executor._discard_task(task_id, tombstone_ttl_seconds)

    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        group_size: int = 1,
    ) -> list[dict[str, Any]]:
        """Submit a batch of requests and wait for results.

        This method does not support asynchronous rollout and should be used for offline
        data collection or debugging, not in production experiments.

        Parameters
        ----------
        data : List[Dict[str, Any]]
            A list of input data dictionaries for rollout
        workflow : WorkflowLike
            The workflow to use for rollout generation
        workflow_kwargs : Dict[str, Any], optional
            Keyword arguments to pass to the workflow constructor
        group_size : int
            Number of times to run the workflow per input and concatenate results.
            Default is 1 (no grouping).

        Returns
        -------
        list[dict[str, Any]]
            A list of trajectory dictionaries, one per accepted rollout result.
            Each trajectory is a dict of tensors with shape [batch_size, seqlen, ...],
            where batch_size can vary per trajectory depending on the workflow output.
        """
        assert workflow is not None, "Workflow must be specified for rollout_batch."

        # Resolve workflow to a RolloutWorkflow instance
        resolved_workflow = self._resolve_workflow(
            workflow, workflow_kwargs, group_size
        )

        return self.workflow_executor.rollout_batch(
            data=data,
            workflow=resolved_workflow,
        )

    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Callable[[dict[str, Any]], bool] | str | None = None,
        group_size: int = 1,
        dynamic_bs: bool = False,
    ) -> list[dict[str, Any]]:
        """Asynchronously submit and wait until a full batch is ready.

        Parameters
        ----------
        dataloader : StatefulDataLoader
            The data loader to pull data from
        workflow : WorkflowLike
            The workflow to use for rollout generation
        workflow_kwargs : Dict[str, Any], optional
            Keyword arguments to pass to the workflow constructor
        should_accept_fn : Callable[[Dict[str, Any]], bool] | str, optional
            A function or module path for trajectory filtering
        group_size : int
            Number of times to run the workflow per input and concatenate results.
            Default is 1 (no grouping).
        dynamic_bs : bool, optional
            If True, enables dynamic batch sizing. Default is False.

        Returns
        -------
        list[dict[str, Any]]
            A list of trajectory dictionaries, one per accepted rollout result.
            Each trajectory is a dict of tensors with shape [batch_size, seqlen, ...],
            where batch_size can vary per trajectory depending on the workflow output.
        """
        assert workflow is not None, "Workflow must be specified for prepare_batch."

        # Resolve workflow to a RolloutWorkflow instance
        resolved_workflow = self._resolve_workflow(
            workflow, workflow_kwargs, group_size
        )
        resolved_should_accept_fn = self._resolve_should_accept_fn(should_accept_fn)

        return self.workflow_executor.prepare_batch(
            dataloader=dataloader,
            workflow=resolved_workflow,
            should_accept_fn=resolved_should_accept_fn,
            dynamic_bs=dynamic_bs,
        )

    @trace_perf("remote_inf_engine.pause_generation", category="misc")
    def pause_generation(self):
        """Hard-pause generation requests to the inference backend."""
        self._set_generation_paused(True)
        try:
            pause_req = self.backend.get_pause_request()
            self._run_request_on_all_servers(pause_req)
        except Exception:
            self._set_generation_paused(False)
            raise

        # The above http request may require some time to be scheduled and executed.
        # The following line waits until all requests are indeed dropped.
        time.sleep(self.config.pause_grace_period)

    def _resume_generation(self):
        try:
            resume_req = self.backend.get_resume_request()
            self._run_request_on_all_servers(resume_req)
        finally:
            self._set_generation_paused(False)

    @trace_perf("remote_inf_engine.resume_generation", category="misc")
    def resume_generation(self):
        """Resume backend generation requests after :meth:`pause_generation`."""
        self._resume_generation()

    @trace_perf("remote_inf_engine.continue_generation", category="misc")
    def continue_generation(self):
        """Compatibility alias for :meth:`resume_generation`."""
        self._resume_generation()

    def pause_rollout_submission(self):
        """Soft-pause starting new rollout workflows.

        Existing workflows may continue and may still call :meth:`agenerate`
        unless :meth:`pause_generation` is used separately.
        """
        return self.workflow_executor.pause_submission()

    def resume_rollout_submission(self):
        """Resume starting new rollout workflows."""
        return self.workflow_executor.resume_submission()

    def pause(self):
        """Compatibility alias for :meth:`pause_rollout_submission`."""
        return self.pause_rollout_submission()

    def resume(self):
        """Compatibility alias for :meth:`resume_rollout_submission`."""
        return self.resume_rollout_submission()

    def offload(self) -> None:
        """Offload model memory on all servers."""
        offload_req = self.backend.get_offload_request()
        self._run_request_on_all_servers(offload_req)

    def onload(self, tags: list[str] | None = None) -> None:
        """Onload model memory on all servers."""
        onload_req = self.backend.get_onload_request(tags=tags)
        self._run_request_on_all_servers(onload_req)

    def _run_request_on_all_servers(self, req: HttpRequest):
        async def _fn():
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.request_timeout),
                read_bufsize=1024 * 1024 * 10,
                connector=get_default_connector(),
            ) as session:
                jobs = []
                for addr in self.addresses:
                    jobs.append(
                        arequest_with_retry(
                            session=session,
                            addr=addr,
                            endpoint=req.endpoint,
                            payload=req.payload,
                            method=req.method,
                            max_retries=self.config.request_retries,
                            timeout=self.config.request_timeout,
                        )
                    )
                await asyncio.gather(*jobs)

        uvloop.run(_fn())

    def launch_server(self, server_args: dict[str, Any]) -> LocalInfServerInfo:
        """Launch a local inference server."""
        server_args["host"] = gethostip()
        server_args["port"] = find_free_ports(1, port_range=(20000, 32000))[0]
        process = self.backend.launch_server(server_args)
        address = format_hostport(server_args["host"], server_args["port"])
        server_info = LocalInfServerInfo(
            host=server_args["host"],
            port=server_args["port"],
            process=process,
        )
        try:
            self._wait_for_server(address, process=process)
            self.local_server_processes.append(server_info)
            self._record_lora_server_args(server_args, address)
            if ray.is_initialized():
                # do not return with process for ray as it is not picklable
                return LocalInfServerInfo(
                    host=server_args["host"],
                    port=server_args["port"],
                    process=None,
                )
            return server_info
        except TimeoutError:
            logger.warning(
                f"Launch local server timeouted at {address} after {self.config.setup_timeout}s."
            )
            self._shutdown_one_server(server_info)
            raise

    def _shutdown_one_server(self, server_info: LocalInfServerInfo):
        addr = format_hostport(server_info.host, server_info.port)
        if addr in self.addresses:
            self.addresses.remove(addr)
        with self._lora_cond:
            self._loaded_lora_versions_by_addr.pop(addr, None)
            self._unloading_lora_versions_by_addr.pop(addr, None)
            self._max_loaded_loras_by_addr.pop(addr, None)
            self._lora_cond.notify_all()
        if server_info.process.poll() is not None:
            return
        kill_process_tree(server_info.process.pid, graceful=True)

    def teardown_server(self):
        """Teardown all locally launched servers."""
        for server_info in self.local_server_processes:
            self._shutdown_one_server(server_info)
        self.local_server_processes.clear()


# Helper functions that run in ProcessPoolExecutor


def _update_weights_from_disk(
    backend: RemoteInfBackendProtocol,
    experiment_name: str,
    trial_name: str,
    model_version: int,
    addresses: list[str],
    meta: WeightUpdateMeta,
    request_retries: int,
    request_timeout: float,
):
    """Helper to update weights from disk in a separate process."""

    async def _fn():
        update_name = names.update_weights_from_disk(
            experiment_name, trial_name, model_version
        )
        save_timestamp = float(name_resolve.wait(update_name, timeout=request_timeout))
        load_timestamp = datetime.now().timestamp()

        # Get requests from backend with version for LoRA name
        weight_reqs = backend.build_disk_weight_update_requests(meta)

        # Execute all requests
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=request_timeout),
            read_bufsize=1024 * 1024 * 10,
            connector=get_default_connector(),
        ) as session:
            for http_req in weight_reqs.requests:
                jobs = [
                    arequest_with_retry(
                        session=session,
                        addr=addr,
                        endpoint=http_req.endpoint,
                        payload=http_req.payload,
                        method=http_req.method,
                        max_retries=request_retries,
                        timeout=request_timeout,
                    )
                    for addr in addresses
                ]
                await asyncio.gather(*jobs)

        return load_timestamp - save_timestamp

    return uvloop.run(_fn())


def _init_weights_update_group_remote(
    backend: RemoteInfBackendProtocol,
    meta: WeightUpdateMeta,
    addresses: list[str],
    request_timeout: float,
    xccl_group_ranks: list[int] | None = None,
):
    """Helper to initialize weight update group in a separate process.

    If xccl_group_ranks is provided, it must have the same length as addresses and will be
    used as the per-address rank passed to the backend request builder.
    Otherwise, ranks default to enumerate(addresses).
    """

    if xccl_group_ranks is not None and len(xccl_group_ranks) != len(addresses):
        raise ValueError(
            f"xccl_group_ranks must have the same length as addresses "
            f"(got {len(xccl_group_ranks)} vs {len(addresses)})"
        )

    async def _fn():
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=request_timeout),
            read_bufsize=1024 * 1024 * 10,
            connector=get_default_connector(),
        ) as session:
            jobs = []
            for i, addr in enumerate(addresses):
                xccl_group_rank = (
                    xccl_group_ranks[i] if xccl_group_ranks is not None else i
                )
                http_req = backend.build_init_weights_group_request(
                    addr, xccl_group_rank, meta
                )
                jobs.append(
                    arequest_with_retry(
                        session=session,
                        addr=addr,
                        endpoint=http_req.endpoint,
                        payload=http_req.payload,
                        method=http_req.method,
                        max_retries=1,
                        timeout=request_timeout,
                    )
                )
            await asyncio.gather(*jobs)

    return uvloop.run(_fn())


def _update_weights_from_distributed(
    backend: RemoteInfBackendProtocol,
    meta: WeightUpdateMeta,
    param_specs: list[ParamSpec],
    addresses: list[str],
    request_timeout: float,
):
    """Helper to update weights from distributed memory in a separate process."""

    async def _fn():
        # Get requests from backend
        weight_reqs = backend.build_distributed_weight_update_requests(
            meta, param_specs
        )

        # Execute all requests sequentially (they may have dependencies)
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=request_timeout),
            read_bufsize=1024 * 1024 * 10,
            connector=get_default_connector(),
        ) as session:
            for http_req in weight_reqs.requests:
                jobs = [
                    arequest_with_retry(
                        session=session,
                        addr=addr,
                        endpoint=http_req.endpoint,
                        payload=http_req.payload,
                        method=http_req.method,
                        max_retries=1,
                        timeout=request_timeout,
                    )
                    for addr in addresses
                ]
                await asyncio.gather(*jobs)

    return uvloop.run(_fn())
