import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        import os
        use_cutile = (os.environ.get("NANO_VLLM_USE_CUTILE", "0") == "1")
        self.use_cutile = use_cutile
        self.enforce_eager = config.enforce_eager
        self.cuda_graphs_enabled = False
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        import os
        backend = "gloo" if (os.name == "nt" or self.world_size == 1) else "nccl"
        dist.init_process_group(backend, "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        import os
        self.use_green_contexts = (os.environ.get("NANO_VLLM_USE_GREEN_CONTEXTS", "0") == "1")
        self.green_api_type = None
        self.green_requested_api = os.environ.get("NANO_VLLM_GREEN_CONTEXT_API", "auto").lower()
        if self.green_requested_api not in {"auto", "pytorch", "cuda_core"}:
            print(f"WARNING: Unknown Green Context API '{self.green_requested_api}', using auto.")
            self.green_requested_api = "auto"
        self.green_prefill_sms = int(os.environ.get("NANO_VLLM_PREFILL_SMS", "32"))
        self.green_decode_sms = int(os.environ.get("NANO_VLLM_DECODE_SMS", "16"))
        if self.use_green_contexts:
            errors = []
            if self.green_requested_api in {"auto", "pytorch"}:
                try:
                    from torch.cuda.green_contexts import GreenContext
                    self.ctx_prefill = GreenContext.create(num_sms=self.green_prefill_sms)
                    self.ctx_decode = GreenContext.create(num_sms=self.green_decode_sms)
                    self.green_api_type = "pytorch"
                    print(f"Green Contexts Initialized (PyTorch API): Prefill Context ({self.green_prefill_sms} SMs), Decode Context ({self.green_decode_sms} SMs)")
                except Exception as e:
                    errors.append(f"pytorch={e}")
                    if self.green_requested_api == "pytorch":
                        print(f"WARNING: Green Context initialization failed: {e}. Falling back to default context.")
                        self.use_green_contexts = False
            if self.use_green_contexts and self.green_api_type is None and self.green_requested_api in {"auto", "cuda_core"}:
                try:
                    from cuda.bindings import driver as cuda
                    from cuda.core import Device, ContextOptions, SMResourceOptions
                    cuda.cuInit(0)
                    dev = Device(0)
                    dev.set_current()
                    sm = dev.resources.sm
                    total_sms = sm.sm_count
                    decode_sm = self.green_decode_sms
                    prefill_sm = self.green_prefill_sms if "NANO_VLLM_PREFILL_SMS" in os.environ else max(1, total_sms - decode_sm)
                    if prefill_sm + decode_sm > total_sms:
                        raise RuntimeError(f"requested split {prefill_sm}+{decode_sm} exceeds total SM count {total_sms}")
                    split_layouts = list(sm.split(SMResourceOptions(count=(decode_sm,))))
                    if not split_layouts:
                        raise RuntimeError("single decode partition split returned no layouts")
                    layout = split_layouts[0]
                    if isinstance(layout, (list, tuple)) and len(layout) >= 1:
                        crit_grp = layout[0]
                        long_grp = layout[1] if len(layout) >= 2 else sm
                    else:
                        crit_grp = layout
                        long_grp = sm
                    self.green_split_layout_width = len(layout) if isinstance(layout, (list, tuple)) else 1
                    self.green_prefill_resource_source = "split_remainder" if isinstance(layout, (list, tuple)) and len(layout) >= 2 else "device_sm_fallback"
                    self.green_device = dev
                    self.ctx_prefill = dev.create_context(ContextOptions(resources=[long_grp]))
                    self.ctx_decode = dev.create_context(ContextOptions(resources=[crit_grp]))
                    self.green_prefill_stream = self.ctx_prefill.create_stream()
                    self.green_decode_stream = self.ctx_decode.create_stream()
                    dev.set_current()
                    self.green_api_type = "cuda_core"
                    print(f"Green Contexts Initialized (cuda.core API): Prefill Context ({prefill_sm} SMs), Decode Context ({decode_sm} SMs)")
                except Exception as ex:
                    errors.append(f"cuda_core={ex}")
                    print(f"WARNING: Green Context initialization failed: {'; '.join(errors)}. Falling back to default context.")
                    self.use_green_contexts = False
        self._alloc_persistent_buffers()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            try:
                self.capture_cudagraph()
                self.cuda_graphs_enabled = True
                if self.use_cutile:
                    print("cuTile CUDA Graph decode capture enabled")
            except Exception as exc:
                if self.use_cutile:
                    print(f"WARNING: cuTile CUDA Graph capture failed; falling back to eager decode: {exc}")
                    self.enforce_eager = True
                else:
                    raise
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if self.cuda_graphs_enabled:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def _alloc_persistent_buffers(self):
        config = self.config
        self._max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        self._buf_input_ids     = torch.empty(config.max_num_batched_tokens, dtype=torch.int64, device="cuda")
        self._buf_positions     = torch.empty(config.max_num_batched_tokens, dtype=torch.int64, device="cuda")
        self._buf_slot_mapping  = torch.empty(config.max_num_batched_tokens, dtype=torch.int32, device="cuda")
        self._buf_cu_seqlens_q  = torch.empty(config.max_num_seqs + 1, dtype=torch.int32, device="cuda")
        self._buf_cu_seqlens_k  = torch.empty(config.max_num_seqs + 1, dtype=torch.int32, device="cuda")
        self._buf_context_lens  = torch.empty(config.max_num_seqs, dtype=torch.int32, device="cuda")
        self._buf_block_tables  = torch.empty(config.max_num_seqs, self._max_num_blocks, dtype=torch.int32, device="cuda")
        self._buf_temperatures  = torch.empty(config.max_num_seqs, dtype=torch.float32, device="cuda")
        self._buf_outputs       = torch.empty(config.max_num_seqs, config.hf_config.hidden_size, dtype=config.hf_config.torch_dtype, device="cuda")
        # Initialize to safe values (token 0, slot -1, context 0, block -1) so warmup /
        # CUDA-graph capture never reads uninitialized memory.
        self._buf_input_ids.zero_()
        self._buf_positions.zero_()
        self._buf_slot_mapping.fill_(-1)
        self._buf_cu_seqlens_q.zero_()
        self._buf_cu_seqlens_k.zero_()
        self._buf_context_lens.zero_()
        self._buf_block_tables.fill_(-1)
        self._buf_temperatures.zero_()
        self._buf_outputs.zero_()

    def _h2d(self, dst: torch.Tensor, values: list, dtype) -> torch.Tensor:
        """Copy a Python list into a preallocated CUDA buffer slice (no per-step allocation)."""
        n = len(values)
        dst[:n].copy_(torch.tensor(values, dtype=dtype, device="cpu"), non_blocking=True)
        return dst[:n]

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        rows = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        cpu = torch.tensor(rows, dtype=torch.int32, device="cpu")
        self._buf_block_tables[:len(seqs), :max_len].copy_(cpu, non_blocking=True)
        return self._buf_block_tables[:len(seqs), :max_len]

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(range(seq.num_cached_tokens, seqlen))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(range(start, end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = self._h2d(self._buf_input_ids, input_ids, torch.int64)
        positions = self._h2d(self._buf_positions, positions, torch.int64)
        cu_seqlens_q = self._h2d(self._buf_cu_seqlens_q, cu_seqlens_q, torch.int32)
        cu_seqlens_k = self._h2d(self._buf_cu_seqlens_k, cu_seqlens_k, torch.int32)
        slot_mapping = self._h2d(self._buf_slot_mapping, slot_mapping, torch.int32)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables, use_cutile=self.use_cutile)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        input_ids = self._h2d(self._buf_input_ids, input_ids, torch.int64)
        positions = self._h2d(self._buf_positions, positions, torch.int64)
        slot_mapping = self._h2d(self._buf_slot_mapping, slot_mapping, torch.int32)
        context_lens = self._h2d(self._buf_context_lens, context_lens, torch.int32)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables, use_cutile=self.use_cutile)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        return self._h2d(self._buf_temperatures, temperatures, torch.float32)

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            cap_bs = next(x for x in self.graph_bs if x >= bs)
            graph = self.graphs[cap_bs]
            # Persistent buffers are filled directly by prepare_decode/prepare_prefill.
            # Only the padding tail (bs..cap_bs) must be reset so inactive entries are skipped.
            if bs < cap_bs:
                self._buf_slot_mapping[bs:cap_bs].fill_(-1)
                self._buf_context_lens[bs:cap_bs].zero_()
            graph.replay()
            return self.model.compute_logits(self._buf_outputs[:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        if self.use_green_contexts:
            ctx = self.ctx_prefill if is_prefill else self.ctx_decode
            if self.green_api_type == "pytorch":
                ctx.set_context()
            elif self.green_api_type == "cuda_core":
                self.green_device.set_current(ctx)
        
        try:
            input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
            temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
            logits = self.run_model(input_ids, positions, is_prefill)
            token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
            reset_context()
            return token_ids
        finally:
            if self.use_green_contexts:
                ctx = self.ctx_prefill if is_prefill else self.ctx_decode
                if self.green_api_type == "pytorch":
                    ctx.pop_context()
                elif self.green_api_type == "cuda_core":
                    self.green_device.set_current()

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        max_bs = min(config.max_num_seqs, 512)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(
                False,
                slot_mapping=self._buf_slot_mapping[:bs],
                context_lens=self._buf_context_lens[:bs],
                block_tables=self._buf_block_tables[:bs],
                use_cutile=self.use_cutile,
            )
            self._buf_outputs[:bs] = self.model(self._buf_input_ids[:bs], self._buf_positions[:bs])  # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                self._buf_outputs[:bs] = self.model(self._buf_input_ids[:bs], self._buf_positions[:bs])  # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()



