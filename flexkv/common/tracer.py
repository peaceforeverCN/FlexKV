import json
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional, List, Union
import torch
import numpy as np

from flexkv.common.config import CacheConfig


class FlexKVTracer:
    """FlexKV Tracer class for recording operations in JSON format"""

    def __init__(self, cache_config: CacheConfig):
        self.enabled = cache_config.enable_trace
        if not self.enabled:
            return
        print(f"FlexKVTracer enabled, trace_file_path: {cache_config.trace_file_path}")
        self.trace_file_path = cache_config.trace_file_path
        self.max_file_size_mb = cache_config.trace_max_file_size_mb
        self.max_files = cache_config.trace_max_files
        self.flush_interval_ms = cache_config.trace_flush_interval_ms

        # Thread-safe file writing
        self._lock = threading.Lock()
        self._buffer = []
        self._last_flush_time = time.time()

        # Create trace file
        self._init_trace_file()

    def _init_trace_file(self):
        """Initialize trace file and create directory if needed"""
        if not self.enabled:
            return

        os.makedirs(os.path.dirname(self.trace_file_path), exist_ok=True)

        # Rotate files if needed
        self._rotate_files_if_needed()

    def _rotate_files_if_needed(self):
        """Rotate trace files if current file is too large"""
        if not os.path.exists(self.trace_file_path):
            return

        file_size_mb = os.path.getsize(self.trace_file_path) / (1024 * 1024)
        if file_size_mb >= self.max_file_size_mb:
            # Rotate files
            for i in range(self.max_files - 1, 0, -1):
                old_file = f"{self.trace_file_path}.{i}"
                new_file = f"{self.trace_file_path}.{i+1}"
                if os.path.exists(old_file):
                    if i == self.max_files - 1:
                        os.remove(old_file)
                    else:
                        os.rename(old_file, new_file)

            # Move current file to .1
            if os.path.exists(self.trace_file_path):
                os.rename(self.trace_file_path, f"{self.trace_file_path}.1")

    def _convert_tensor_to_list(self, obj: Any) -> Any:
        """Convert torch tensors and numpy arrays to lists for JSON serialization"""
        if isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_tensor_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_tensor_to_list(item) for item in obj]
        else:
            return obj

    def _write_to_file(self, json_record: str):
        """Write JSON record to file"""
        with open(self.trace_file_path, 'a', encoding='utf-8') as f:
            f.write(json_record + '\n')

    def _flush_buffer(self):
        """Flush buffered records to file"""
        if not self._buffer:
            return

        if self._buffer:
            records = self._buffer.copy()
            self._buffer.clear()

            for record in records:
                self._write_to_file(record)

            self._last_flush_time = time.time()

    def trace_config(self, model_config, cache_config, gpu_layout=None):
        """Record system configuration"""
        if not self.enabled:
            return

        timestamp = datetime.now().isoformat()

        # Convert model_config to dict
        model_config_dict = {
            "num_layers": model_config.num_layers,
            "num_kv_heads": model_config.num_kv_heads,
            "head_size": model_config.head_size,
            "use_mla": model_config.use_mla,
            "dtype": str(model_config.dtype),
            "tp_size": model_config.tp_size,
            "dp_size": model_config.dp_size,
        }

        # Convert cache_config to dict
        cache_config_dict = {
            "tokens_per_block": cache_config.tokens_per_block,
            "enable_cpu": cache_config.enable_cpu,
            "enable_ssd": cache_config.enable_ssd,
            "enable_remote": cache_config.enable_remote,
            "gpu_kv_layout_type": str(cache_config.gpu_kv_layout_type),
            "cpu_kv_layout_type": str(cache_config.cpu_kv_layout_type),
            "ssd_kv_layout_type": str(cache_config.ssd_kv_layout_type),
            "remote_kv_layout_type": str(cache_config.remote_kv_layout_type),
            "use_gds": cache_config.use_gds,
            "remote_cache_size_mode": cache_config.remote_cache_size_mode,
            "num_cpu_blocks": cache_config.num_cpu_blocks,
            "num_ssd_blocks": cache_config.num_ssd_blocks,
            "num_remote_blocks": cache_config.num_remote_blocks,
            "remote_file_size": cache_config.remote_file_size,
            "remote_file_num": cache_config.remote_file_num,
            "remote_file_prefix": cache_config.remote_file_prefix,
            "ssd_cache_dir": cache_config.ssd_cache_dir,
            "ssd_cache_iouring_entries": cache_config.ssd_cache_iouring_entries,
            "ssd_cache_iouring_flags": cache_config.ssd_cache_iouring_flags,
            "remote_cache_path": cache_config.remote_cache_path,
            "remote_config_custom": cache_config.remote_config_custom,
            "evict_ratio": cache_config.evict_ratio,
        }

        # Convert gpu_layout to dict if provided
        gpu_layout_dict = None
        if gpu_layout is not None:
            gpu_layout_dict = {
                "type": str(gpu_layout.type),
                "num_layer": gpu_layout.num_layer,
                "num_block": gpu_layout.num_block,
                "tokens_per_block": gpu_layout.tokens_per_block,
                "num_head": gpu_layout.num_head,
                "head_size": gpu_layout.head_size,
                "is_mla": gpu_layout.is_mla,
            }

        record = {
            "timestamp": timestamp,
            "event_type": "config",
            "component": "KVManager",
            "data": {
                "model_config": model_config_dict,
                "cache_config": cache_config_dict,
                "gpu_layout": gpu_layout_dict,
            }
        }

        json_record = json.dumps(record, ensure_ascii=False, separators=(',', ':'))

        with self._lock:
            self._buffer.append(json_record)

            # Check if we need to flush
            current_time = time.time()
            if (current_time - self._last_flush_time) * 1000 >= self.flush_interval_ms:
                self._flush_buffer()

    def trace_request(self,
                     request_type: str,
                     request_id: int,
                     token_ids: torch.Tensor,
                     slot_mapping: torch.Tensor,
                     token_mask: Optional[torch.Tensor] = None,
                     layer_granularity: int = -1,
                     dp_id: int = 0,
                     **kwargs):
        """Record a request operation"""
        if not self.enabled:
            return

        timestamp = datetime.now().isoformat()

        # Convert tensors to lists for JSON serialization
        data = {
            "request_type": request_type,
            "request_id": request_id,
            "token_ids": self._convert_tensor_to_list(token_ids),
            "slot_mapping": self._convert_tensor_to_list(slot_mapping),
            "token_mask": self._convert_tensor_to_list(token_mask) if token_mask is not None else None,
            "layer_granularity": layer_granularity,
            "dp_id": dp_id,
            "token_ids_shape": list(token_ids.shape),
            "slot_mapping_shape": list(slot_mapping.shape),
            "token_mask_shape": list(token_mask.shape) if token_mask is not None else None,
        }

        # Add any additional kwargs
        for key, value in kwargs.items():
            data[key] = self._convert_tensor_to_list(value)

        record = {
            "timestamp": timestamp,
            "event_type": "request",
            "component": "KVManager",
            "data": data
        }
        json_record = json.dumps(record, ensure_ascii=False, separators=(',', ':'))
        with self._lock:
            self._buffer.append(json_record)
            # Check if we need to flush
            current_time = time.time()
            if (current_time - self._last_flush_time) * 1000 >= self.flush_interval_ms:
                self._flush_buffer()

    def trace_wait_request(self,
                          wait_type: str,
                          task_ids: Union[int, List[int]],
                          layer_group_id: Optional[int] = None):
        """Record a wait operation"""
        if not self.enabled:
            return

        timestamp = datetime.now().isoformat()

        # Convert task_ids to list if it's a single int
        if isinstance(task_ids, int):
            task_ids_list = [task_ids]
        else:
            task_ids_list = list(task_ids)

        data = {
            "wait_type": wait_type,
            "task_ids": task_ids_list,
            "layer_group_id": layer_group_id,
        }
        record = {
            "timestamp": timestamp,
            "event_type": "wait",
            "component": "KVManager",
            "data": data
        }

        json_record = json.dumps(record, ensure_ascii=False, separators=(',', ':'))

        with self._lock:
            self._buffer.append(json_record)

            # Check if we need to flush
            current_time = time.time()
            if (current_time - self._last_flush_time) * 1000 >= self.flush_interval_ms:
                self._flush_buffer()

    def flush(self):
        """Manually flush all buffered records"""
        if not self.enabled:
            return

        self._flush_buffer()

    def __del__(self):
        """Ensure all records are flushed when tracer is destroyed"""
        from contextlib import suppress
        with suppress(Exception):
            self.flush()
