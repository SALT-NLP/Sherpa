# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import dataclasses
import json
import os
import pickle
import shutil
import time
from typing import TYPE_CHECKING, Any

import torch.distributed as dist
from transformers import PreTrainedTokenizerFast

if TYPE_CHECKING:
    from transformers import AutoProcessor

from areal.api import (
    FinetuneSpec,
    InferenceEngine,
    SaveLoadMeta,
    StepInfo,
    TrainEngine,
    WeightUpdateMeta,
)
from areal.api.cli_args import RecoverConfig
from areal.infra import TrainController
from areal.utils import logging, timeutil
from areal.utils.evaluator import Evaluator
from areal.utils.saver import Saver

if TYPE_CHECKING:
    from areal.utils.stats_logger import StatsLogger

logger = logging.getLogger("Recover")


class InValidRecoverInfo(Exception):
    pass


@dataclasses.dataclass
class RecoverInfo:
    # Last step info is the counter of the saved checkpoint.
    # Recover will start from the next iteration, obtained by `last_step_info.next()`.
    last_step_info: StepInfo

    saver_info: dict
    evaluator_info: dict
    stats_logger_info: dict
    dataloader_info: dict | list[dict]
    checkpoint_info: dict

    def dump(self, dump_dir: str):
        # Dumps the recover info to multiple files in `dump_dir`:
        # 1. step_info.json: contains the recover info
        # 2. *_info.json or *_info.pkl: contains other informantion required for recover.

        if dist.is_initialized():
            # Since dataloader state is different across distributed ranks,
            # we need to all gather the dataloader state from all ranks.
            # In this situation, saved dataloader_info is a list of states from all ranks.
            dataloader_info = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(dataloader_info, self.dataloader_info)

            # To avoid contention, do not dump on multiple ranks
            if dist.get_rank() != 0:
                return
        else:
            dataloader_info = self.dataloader_info

        os.makedirs(dump_dir, exist_ok=True)
        step_info_path = os.path.join(dump_dir, "step_info.json")
        with open(step_info_path, "w") as f:
            json.dump(dataclasses.asdict(self.last_step_info), f, indent=4)

        saver_info_path = os.path.join(dump_dir, "saver_info.json")
        with open(saver_info_path, "w") as f:
            json.dump(self.saver_info, f, indent=4)

        evaluator_info_path = os.path.join(dump_dir, "evaluator_info.json")
        with open(evaluator_info_path, "w") as f:
            json.dump(self.evaluator_info, f, indent=4)

        stats_logger_info_path = os.path.join(dump_dir, "stats_logger_info.json")
        with open(stats_logger_info_path, "w") as f:
            json.dump(self.stats_logger_info, f, indent=4)

        checkpoint_info_path = os.path.join(dump_dir, "checkpoint_info.json")
        with open(checkpoint_info_path, "w") as f:
            json.dump(self.checkpoint_info, f, indent=4)

        dataloader_info_path = os.path.join(dump_dir, "dataloader_info.pkl")
        with open(dataloader_info_path, "wb") as f:
            pickle.dump(dataloader_info, f)

    @classmethod
    def load(cls, load_dir: str):
        # Loads the recover info from multiple files in `load_dir`:
        if not os.path.exists(load_dir):
            raise FileNotFoundError(
                f"Recover info directory {load_dir} does not exist."
            )

        try:
            step_info_path = os.path.join(load_dir, "step_info.json")
            with open(step_info_path) as f:
                step_info_dict = json.load(f)
                last_step_info = StepInfo(**step_info_dict)

            evaluator_info_path = os.path.join(load_dir, "evaluator_info.json")
            with open(evaluator_info_path) as f:
                evaluator_info = json.load(f)

            saver_info_path = os.path.join(load_dir, "saver_info.json")
            with open(saver_info_path) as f:
                saver_info = json.load(f)

            stats_logger_info_path = os.path.join(load_dir, "stats_logger_info.json")
            with open(stats_logger_info_path) as f:
                stats_logger_info = json.load(f)

            checkpoint_info_path = os.path.join(load_dir, "checkpoint_info.json")
            with open(checkpoint_info_path) as f:
                checkpoint_info = json.load(f)

            dataloader_info_path = os.path.join(load_dir, "dataloader_info.pkl")
            with open(dataloader_info_path, "rb") as f:
                dataloader_info = pickle.load(f)
                if isinstance(dataloader_info, list):
                    # If dataloader_info a list, it means it is saved from a distributed run.
                    if dist.is_initialized():
                        # Loading dataloader states in a distributed context.
                        assert dist.get_world_size() == len(dataloader_info), (
                            f"Dataloader info list length {len(dataloader_info)} does not match "
                            f"the world size {dist.get_world_size()}."
                        )
                        dataloader_info = dataloader_info[dist.get_rank()]

            return cls(
                last_step_info=last_step_info,
                saver_info=saver_info,
                evaluator_info=evaluator_info,
                stats_logger_info=stats_logger_info,
                dataloader_info=dataloader_info,
                checkpoint_info=checkpoint_info,
            )
        except Exception as e:
            logger.error(f"Failed to load recover info from {load_dir}: {e}")
            raise InValidRecoverInfo(f"Invalid recover info in {load_dir}") from e


class RecoverHandler:
    _RECOVER_DIR = "recover"
    _GENERATIONS_DIR = "generations"
    _CURRENT_FILE = "current.json"

    def __init__(self, config: RecoverConfig, ft_spec: FinetuneSpec):
        self.config = config
        self.ft_spec = ft_spec
        self.last_step_info = StepInfo(
            epoch=-1,
            epoch_step=-1,
            global_step=-1,
            steps_per_epoch=ft_spec.steps_per_epoch,
        )
        self.freq_ctl = timeutil.EpochStepTimeFreqCtl(
            freq_epoch=config.freq_epochs,
            freq_step=config.freq_steps,
            freq_sec=config.freq_secs,
        )

    @staticmethod
    def recover_info_path(
        experiment_name: str,
        trial_name: str,
        fileroot: str,
    ):
        return os.path.join(
            Saver.get_save_root(experiment_name, trial_name, fileroot),
            "recover_info",
        )

    @classmethod
    def _recover_root(cls, experiment_name: str, trial_name: str, fileroot: str) -> str:
        return os.path.join(
            Saver.get_save_root(experiment_name, trial_name, fileroot),
            cls._RECOVER_DIR,
        )

    @classmethod
    def _generation_path(
        cls,
        experiment_name: str,
        trial_name: str,
        fileroot: str,
        generation: str,
    ) -> str:
        cls._validate_path_component(generation, label="recovery generation")
        return os.path.join(
            cls._recover_root(experiment_name, trial_name, fileroot),
            cls._GENERATIONS_DIR,
            generation,
        )

    @classmethod
    def _read_current_generation(
        cls, experiment_name: str, trial_name: str, fileroot: str
    ) -> str:
        pointer_path = os.path.join(
            cls._recover_root(experiment_name, trial_name, fileroot),
            cls._CURRENT_FILE,
        )
        with open(pointer_path) as f:
            generation = json.load(f)["generation"]
        cls._validate_path_component(generation, label="recovery generation")
        return generation

    @staticmethod
    def _validate_path_component(value: Any, *, label: str) -> None:
        if (
            not isinstance(value, str)
            or not value
            or value in {".", ".."}
            or os.path.basename(value) != value
        ):
            raise InValidRecoverInfo(f"Invalid {label}: {value!r}")

    @classmethod
    def _commit_generation(
        cls,
        experiment_name: str,
        trial_name: str,
        fileroot: str,
        generation: str,
    ) -> bool:
        cls._validate_path_component(generation, label="recovery generation")
        recover_root = cls._recover_root(experiment_name, trial_name, fileroot)
        os.makedirs(recover_root, exist_ok=True)
        pointer_path = os.path.join(recover_root, cls._CURRENT_FILE)
        temporary_path = f"{pointer_path}.tmp-{os.getpid()}"
        try:
            with open(temporary_path, "w") as f:
                json.dump({"generation": generation}, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary_path, pointer_path)
            try:
                directory_fd = os.open(recover_root, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError as exc:
                logger.warning(
                    "Recovery pointer was replaced, but its directory could not "
                    "be synced; keeping the previous generation: %s",
                    exc,
                )
                return False
            return True
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)

    @staticmethod
    def _sync_stage_error(stage: str, error: Exception | None) -> None:
        if not dist.is_initialized():
            if error is not None:
                raise error
            return

        local_error = (
            None
            if error is None
            else f"rank {dist.get_rank()}: {type(error).__name__}: {error}"
        )
        errors: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(errors, local_error)
        failures = [item for item in errors if item is not None]
        if failures:
            raise RuntimeError(f"Recovery {stage} failed: {'; '.join(failures)}")

    @staticmethod
    def _new_generation_name(step_info: StepInfo) -> str:
        timestamp = (
            time.time_ns()
            if not dist.is_initialized() or dist.get_rank() == 0
            else None
        )
        if dist.is_initialized():
            values = [timestamp]
            dist.broadcast_object_list(values, src=0)
            timestamp = values[0]
        return (
            f"epoch{step_info.epoch}-epochstep{step_info.epoch_step}-"
            f"globalstep{step_info.global_step}-{timestamp}"
        )

    @staticmethod
    def _cleanup_failed_generation(path: str) -> None:
        if not dist.is_initialized() or dist.get_rank() == 0:
            shutil.rmtree(path, ignore_errors=True)

    @classmethod
    def _cleanup_old_generations(
        cls,
        experiment_name: str,
        trial_name: str,
        fileroot: str,
        current_generation: str,
        keep_last_n: int,
    ) -> None:
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        try:
            generations_root = os.path.join(
                cls._recover_root(experiment_name, trial_name, fileroot),
                cls._GENERATIONS_DIR,
            )
            if not os.path.isdir(generations_root):
                return

            previous_generations: list[tuple[int, str, str]] = []
            invalid_generations: list[tuple[str, str]] = []
            for name in os.listdir(generations_root):
                if name == current_generation:
                    continue
                path = os.path.join(generations_root, name)
                if not os.path.isdir(path):
                    continue
                step_info_path = os.path.join(path, "recover_info", "step_info.json")
                try:
                    with open(step_info_path) as f:
                        step_info = json.load(f)
                    timestamp = int(name.rsplit("-", 1)[-1])
                    int(step_info["global_step"])
                except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                    invalid_generations.append((name, path))
                    continue
                previous_generations.append((timestamp, name, path))

            previous_generations.sort(reverse=True)
            retained_previous = {
                name for _, name, _ in previous_generations[: keep_last_n - 1]
            }
            removed = 0
            for name, path in invalid_generations:
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
            for _, name, path in previous_generations:
                if name in retained_previous:
                    continue
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
            logger.info(
                "Retained %d recovery generation(s) (keep_last_n=%d); "
                "removed %d old or invalid generation(s).",
                1 + len(retained_previous),
                keep_last_n,
                removed,
            )
        except Exception as exc:
            logger.warning("Failed to clean old recovery generations: %s", exc)

    @staticmethod
    def _is_gateway_train_controller(
        engine: TrainEngine
        | TrainController
        | dict[str, TrainEngine | TrainController],
    ) -> bool:
        from areal.experimental.training_service.controller.controller import (
            GatewayTrainController,
        )

        if isinstance(engine, GatewayTrainController):
            return True
        if isinstance(engine, dict):
            return any(
                isinstance(controller, GatewayTrainController)
                for controller in engine.values()
            )
        return False

    def _ensure_recover_supported(
        self,
        engine: TrainEngine
        | TrainController
        | dict[str, TrainEngine | TrainController],
    ) -> None:
        if self._is_gateway_train_controller(engine):
            raise NotImplementedError(
                "Recovery is not supported with GatewayTrainController "
                '(`_version="v2"`) yet. Disable `recover.mode` or use '
                '`_version="v1"`.'
            )

    @staticmethod
    def _normalize_recover_engines(
        engine: TrainEngine
        | TrainController
        | dict[str, TrainEngine | TrainController],
    ) -> dict[str, TrainEngine | TrainController]:
        if isinstance(engine, dict):
            return engine
        return {"default": engine}

    def dump(
        self,
        engine: TrainEngine
        | TrainController
        | dict[str, TrainEngine | TrainController],
        step_info: StepInfo,
        saver: Saver,
        evaluator: Evaluator,
        stats_logger: StatsLogger,
        dataloader: Any,
        tokenizer: PreTrainedTokenizerFast | None = None,
        processor: AutoProcessor | None = None,
        base_model_path: str | None = None,
    ):
        if self.config.mode in ("disabled", "off"):
            return
        self._ensure_recover_supported(engine)
        # currently only support recover on one engine
        if not self.freq_ctl.check(
            epochs=int(step_info.epoch_step == self.ft_spec.steps_per_epoch - 1),
            steps=1,
        ):
            return
        normalized_engine: dict[str, TrainEngine | TrainController] = (
            self._normalize_recover_engines(engine)
        )
        engine_name_error = None
        try:
            for name in normalized_engine:
                self._validate_path_component(name, label="recovery engine name")
        except InValidRecoverInfo as exc:
            engine_name_error = exc
        self._sync_stage_error("engine name validation", engine_name_error)
        generation = self._new_generation_name(step_info)
        generation_path = self._generation_path(
            self.config.experiment_name,
            self.config.trial_name,
            self.config.fileroot,
            generation,
        )
        checkpoint_root = os.path.join(generation_path, "checkpoints")
        for name, engine_ in normalized_engine.items():
            checkpoint_error = None
            try:
                self._save_checkpoint(
                    engine_,
                    path=os.path.join(checkpoint_root, name),
                    tokenizer=tokenizer,
                    processor=processor,
                    base_model_path=base_model_path,
                )
            except Exception as exc:
                checkpoint_error = exc
            try:
                self._sync_stage_error(f"checkpoint save for {name}", checkpoint_error)
            except Exception:
                self._cleanup_failed_generation(generation_path)
                raise

        recover_info = None
        state_collection_error = None
        try:
            recover_info = RecoverInfo(
                last_step_info=step_info,
                saver_info=saver.state_dict(),
                evaluator_info=evaluator.state_dict(),
                stats_logger_info=stats_logger.state_dict(),
                dataloader_info=dataloader.state_dict(),
                checkpoint_info=self.freq_ctl.state_dict(),
            )
        except Exception as exc:
            state_collection_error = exc
        try:
            self._sync_stage_error("state collection", state_collection_error)
        except Exception:
            self._cleanup_failed_generation(generation_path)
            raise
        assert recover_info is not None

        recover_info_path = os.path.join(generation_path, "recover_info")
        info_error = None
        try:
            recover_info.dump(recover_info_path)
        except Exception as exc:
            info_error = exc
        try:
            self._sync_stage_error("state save", info_error)
        except Exception:
            self._cleanup_failed_generation(generation_path)
            raise

        commit_error = None
        commit_is_durable = False
        if not dist.is_initialized() or dist.get_rank() == 0:
            try:
                commit_is_durable = self._commit_generation(
                    self.config.experiment_name,
                    self.config.trial_name,
                    self.config.fileroot,
                    generation,
                )
            except Exception as exc:
                commit_error = exc
        try:
            self._sync_stage_error("commit", commit_error)
        except Exception:
            self._cleanup_failed_generation(generation_path)
            raise
        self.last_step_info = step_info
        if commit_is_durable:
            self._cleanup_old_generations(
                self.config.experiment_name,
                self.config.trial_name,
                self.config.fileroot,
                generation,
                self.config.keep_last_n,
            )

    def load(
        self,
        engine: TrainEngine | dict[str, TrainEngine] | TrainController,
        saver: Saver,
        evaluator: Evaluator,
        stats_logger: StatsLogger,
        dataloader: Any,
        inference_engine: InferenceEngine | None = None,
        weight_update_meta: WeightUpdateMeta | None = None,
        inference_engine_update_from: str = "default",
    ) -> RecoverInfo | None:
        if self.config.mode in ("disabled", "off"):
            return
        self._ensure_recover_supported(engine)
        if inference_engine is not None and weight_update_meta is None:
            raise ValueError("Weight update meta is required for recovery.")

        # TODO(agent): GatewayTrainController is currently duck-typed and does
        # not satisfy this TrainController type check. Extend recovery to accept
        # controller-v2 instances (or make v2 inherit TrainController) before
        # relying on resumed runs with `_version="v2"`.
        normalized_engine: dict[str, TrainEngine | TrainController] = (
            self._normalize_recover_engines(engine)
        )

        generation_path = None
        recover_info = None
        read_error = None
        try:
            generation = self._read_current_generation(
                self.config.experiment_name,
                self.config.trial_name,
                self.config.fileroot,
            )
            generation_path = self._generation_path(
                self.config.experiment_name,
                self.config.trial_name,
                self.config.fileroot,
                generation,
            )
            recover_info_path = os.path.join(generation_path, "recover_info")
            logger.info(f"Loading recover info from {recover_info_path}")
            recover_info = RecoverInfo.load(recover_info_path)
        except Exception as exc:
            read_error = exc
        if isinstance(read_error, FileNotFoundError):
            logger.warning("No recovery state was found. Starting a new experiment.")
            return None
        if read_error is not None and not dist.is_initialized():
            if isinstance(
                read_error,
                (
                    FileNotFoundError,
                    KeyError,
                    TypeError,
                    json.JSONDecodeError,
                    InValidRecoverInfo,
                ),
            ):
                logger.warning(
                    "Complete resume info was not found. "
                    "This should not be a resumed experiment!"
                )
                return None
            raise read_error
        self._sync_stage_error("state read", read_error)

        assert generation_path is not None
        assert recover_info is not None
        logger.info(f"Recovering from {recover_info.last_step_info.next()}.")

        checkpoint_paths: dict[str, str] = {}
        path_error = None
        try:
            for name in normalized_engine:
                self._validate_path_component(name, label="recovery engine name")
                checkpoint_path = os.path.join(generation_path, "checkpoints", name)
                if not os.path.exists(checkpoint_path):
                    raise FileNotFoundError(
                        f"Checkpoint path {checkpoint_path} does not exist."
                    )
                checkpoint_paths[name] = checkpoint_path
        except Exception as exc:
            path_error = exc
        if path_error is not None and not dist.is_initialized():
            logger.warning(
                "Complete resume info was not found. "
                "This should not be a resumed experiment!"
            )
            return None
        self._sync_stage_error("checkpoint validation", path_error)

        state_restore_error = None
        try:
            saver.load_state_dict(recover_info.saver_info)
            self.freq_ctl.load_state_dict(recover_info.checkpoint_info)
            evaluator.load_state_dict(recover_info.evaluator_info)
            stats_logger.load_state_dict(recover_info.stats_logger_info)
            dataloader.load_state_dict(recover_info.dataloader_info)
        except Exception as exc:
            state_restore_error = exc
        self._sync_stage_error("local state restore", state_restore_error)

        for name, engine_ in normalized_engine.items():
            self._load_checkpoint(engine_, path=checkpoint_paths[name])

        global_step = recover_info.last_step_info.global_step

        if inference_engine is not None:
            assert weight_update_meta is not None
            update_engine = normalized_engine[inference_engine_update_from]
            recovery_version = global_step + 1
            versioned_meta = weight_update_meta.with_version(recovery_version)
            update_engine.connect_engine(inference_engine, versioned_meta)
            inference_engine.pause()
            try:
                update_engine.update_weights(versioned_meta)
            finally:
                inference_engine.resume()
            update_engine.set_version(recovery_version)
            inference_engine.set_version(recovery_version)
        return recover_info

    def _save_checkpoint(
        self,
        engine: TrainEngine,
        path: str,
        tokenizer: PreTrainedTokenizerFast | None = None,
        processor: AutoProcessor | None = None,
        base_model_path: str | None = None,
    ):
        os.makedirs(path, exist_ok=True)
        weight_format = "dcp"
        with_optim = not self.config.no_save_optim
        meta = SaveLoadMeta(
            path=path,
            weight_format=weight_format,
            with_optim=with_optim,
            tokenizer=tokenizer,
            processor=processor,
            base_model_path=base_model_path,
        )
        engine.save(meta)
        logger.info(f"Saved recover checkpoint to {path} (with_optim={with_optim})")

    def _load_checkpoint(
        self,
        engine: TrainEngine | TrainController,
        path: str,
        tokenizer: PreTrainedTokenizerFast | None = None,
        base_model_path: str | None = None,
    ):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint path {path} does not exist.")
        weight_format = "dcp"
        with_optim = not self.config.no_load_optim
        meta = SaveLoadMeta(
            path=path,
            weight_format=weight_format,
            with_optim=with_optim,
            tokenizer=None,
            processor=None,
            base_model_path=None,
        )
        engine.load(meta)


def check_if_auto_recover(config: RecoverConfig) -> bool:
    # This method is called by check_if_recover to check if the experiment should
    # recover from a previous run when recovery is enabled ("on" or "auto" mode).
    experiment_name = config.experiment_name
    trial_name = config.trial_name
    fileroot = config.fileroot
    try:
        generation = RecoverHandler._read_current_generation(
            experiment_name, trial_name, fileroot
        )
        generation_path = RecoverHandler._generation_path(
            experiment_name, trial_name, fileroot, generation
        )
        recover_info_path = os.path.join(generation_path, "recover_info")
        logger.info(f"Searching for recover info file in {recover_info_path}.")
        info = RecoverInfo.load(recover_info_path)
        if info.last_step_info.epoch < 0:
            msg = (
                f"Recover checkpoint is not valid. "
                f"Expected last_step_info.epoch >= 0, "
                f"but found {info.last_step_info.epoch}"
            )
            logger.warning(msg)
            return False

        checkpoint_root = os.path.join(generation_path, "checkpoints")
        if not os.path.isdir(checkpoint_root) or not os.listdir(checkpoint_root):
            logger.warning("Recovery checkpoint directory is missing or empty.")
            return False
        return True
    except Exception as e:
        logger.warning(f"Complete recovery state was not found: {e}")
        return False


def check_if_recover(config: RecoverConfig, _run_id: int) -> bool:
    """Check if the experiment should be a recover run.

    When recovery is enabled ('on' or 'auto'), this checks if valid recover
    info and checkpoints are available for automatic recovery.

    Args:
        config: Recovery configuration.
        _run_id: Unused. Kept for API compatibility.

    Returns:
        True if the experiment should recover from a previous run.
    """
    if config.mode in ("disabled", "off"):
        return False
    # Both "on" and "auto" use auto-recovery behavior
    return check_if_auto_recover(config)
