from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import cudnn
import time
import logging
from dataclasses import dataclass
import math

from sglang.srt.layers.attention import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner



@dataclass
class _CuDNNInputParameters:
    num_heads = 4
    head_size = 128
    max_total_num_tokens = 300
    max_num_reqs = 100
    max_context_lenght = 300


class CuDNNBackend(AttentionBackend):
    @dataclass
    class _ArgMapKeys:
        q = "q"
        k_container = "k_container"
        v_container = "v_container"
        k_page_table = "k_page_table"
        v_page_table = "v_page_table"
        seq_len_q_tensor_info = "seq_len_q_tensor_info"
        seq_len_kv_tensor_info = "seq_len_kv_tensor_info"
        o = "o"

    def __init__(self, model_runner: ModelRunner, extend_seq_len_interval = 64):
        super().__init__()
        self.forward_metadata = None

        self._model_runner=model_runner
        # should the number of requests be max_request or max_batch_size
        self.input_size_params=_CuDNNInputParameters(
            num_heads=model_runner.model_config.num_attention_heads,
            head_size=model_runner.model_config.head_dim,
            max_total_num_tokens=model_runner.max_total_num_tokens,
            max_num_reqs = model_runner.server_args.max_running_requests
        )

        self._extend_seq_len_interval = extend_seq_len_interval
        self._decode_graphs=self._init_decode_graphs()
        self._prefill_graphs=self._init_prefill_graphs()


    def _create_cudnn_graph(self, batch_size:int, query_shape, kv_container_shape, kv_page_table_shape, seq_len_shape,max_seq_len):
        graph = cudnn.pygraph(
                io_data_type=cudnn.data_type.HALF,
                intermediate_data_type=cudnn.data_type.FLOAT,
                compute_data_type=cudnn.data_type.FLOAT,
            )

            # q shape: [num_token, num_heads, 1,  head_size], where 1 is sequence length
        q_cudnn = graph.tensor(
            name="q",
            dim=query_shape,
            stride=self._make_compact_strides(query_shape),
            data_type=cudnn.data_type.HALF,
        )

        # container: num_blocks, num heads, tokens_per_block, dim
        # container: max_tokens, num_heads, 1, head_dim since sglang block size is 1
        k_container_cudnn = graph.tensor(
            name="k_container",
            dim=kv_container_shape,
            stride=self._make_compact_strides(kv_container_shape),
            data_type=cudnn.data_type.HALF,
        )
        v_container_cudnn = graph.tensor(
            name="v_container",
            dim=kv_container_shape,
            stride=self._make_compact_strides(kv_container_shape),
            data_type=cudnn.data_type.HALF,
        )

        k_page_table = graph.tensor(
            name="k_page_table",
            dim=kv_page_table_shape,
            stride=self._make_compact_strides(kv_page_table_shape),
            data_type=cudnn.data_type.HALF,
        )
        v_page_table = graph.tensor(
            name="v_page_table",
            dim=kv_page_table_shape,
            stride=self._make_compact_strides(kv_page_table_shape),
            data_type=cudnn.data_type.HALF,
        )
            
        kv_seq_len = graph.tensor(
            name="kv_seq_len",
            dim=seq_len_shape,
            stride=self._make_compact_strides(seq_len_shape),
            data_type=cudnn.data_type.HALF,
        )
        q_seq_len = graph.tensor(
            name="q_seq_len",
            dim=seq_len_shape,
            stride=self._make_compact_strides(seq_len_shape),
            data_type=cudnn.data_type.HALF,
        )

        # TODO: casual, padding mask and gqa same as forward call
        o, _ = graph.sdpa(
            name="sdpa",
            q=q_cudnn,
            k=k_container_cudnn,  # Container K: non contiguous container with K blocks
            v=v_container_cudnn,  # Container V: non contiguous container with V blocks
            is_inference=True,
            attn_scale=self._model_runner.model_config.scaling,
            use_causal_mask=True,
            use_padding_mask=True,
            seq_len_q=q_seq_len,
            seq_len_kv=kv_seq_len,
            paged_attention_k_table=k_page_table,  # Page Table K: Tensor containing offsets to the container with K blocks
            paged_attention_v_table=v_page_table,  # Page Table V: Tensor containing offsets to the container with V blocks
            paged_attention_max_seq_len_kv=max_seq_len,  # The maximum sequence length for K caches (this is optional, but recommended)
        )
        logging.info(graph)


        o.set_output(True).set_dim(query_shape).set_stride(self._make_compact_strides(query_shape))
        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
        graph.build_plans()

        args_map = {
                self._ArgMapKeys.q: q_cudnn,
                self._ArgMapKeys.k_container: k_container_cudnn,
                self._ArgMapKeys.v_container: v_container_cudnn,
                self._ArgMapKeys.k_page_table: k_page_table,
                self._ArgMapKeys.v_page_table: v_page_table, 
                self._ArgMapKeys.seq_len_q_tensor_info: q_seq_len,
                self._ArgMapKeys.seq_len_kv_tensor_info: kv_seq_len,
                self._ArgMapKeys.o: o,
            }
        return args_map,graph

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        pass


    def _make_compact_strides(tensor_shape):
            """Make compact strides for a tensor shape."""
            strides = []
            stride = 1
            for dim in reversed(tensor_shape):
                strides.append(stride)
                stride *= dim
            return list(reversed(strides))
    
    def _init_decode_graphs(self):
        
        max_seq_len = self._model_runner.server_args.max_total_tokens
        decode_graphs = []
        for batch_size in range(1,max_batch_size+1):
            # Radix Attention use KVCache of Block Size 1

            q_shape=[batch_size, self._model_runner.model_config.num_attention_heads,1,self._model_runner.model_config.head_dim]
            kv_container_shape=[self._model_runner.server_args.max_total_tokens,self._model_runner.model_config.num_attention_heads,1,self._model_runner.model_config.head_dim]
            kv_page_table_shape=[self._model_runner.server_args.max_running_requests,self._model_runner.server_args.max_total_tokens]
            seq_len_shape=[batch_size,1,1,1]

            tensor_args,graph = self._create_cudnn_graph(batch_size, q_shape, kv_container_shape, kv_page_table_shape, seq_len_shape,max_seq_len)
            decode_graphs.append((tensor_args, graph))
            assert batch_size == len(decode_graphs), f"batch size {batch_size} does not match the number of graphs {len(self._decode_graphs)}"

        return decode_graphs

    
    def _init_prefill_graphs(self):
        max_seq_len = self._model_runner.server_args.max_total_tokens
        prefill_graphs = []
        ceil_dev = (max_seq_len+self._extend_seq_len_interval+1)/self._extend_seq_len_interval
        for b in range(1, ceil_dev):
            # Create a new graph for each batch size with step size of extend_seq_len_interval (default 64)
            batch_size = b * self._extend_seq_len_interval

            q_shape=[batch_size, self._model_runner.model_config.num_attention_heads,1,self._model_runner.model_config.head_dim]
            kv_container_shape=[self._model_runner.server_args.max_total_tokens,self._model_runner.model_config.num_attention_heads,1,self._model_runner.model_config.head_dim]
            kv_page_table_shape=[self._model_runner.server_args.max_running_requests,self._model_runner.server_args.max_total_tokens]
            seq_len_shape=[batch_size,1,1,1]

            tensor_args,graph = self._create_cudnn_graph(batch_size, q_shape, kv_container_shape, kv_page_table_shape, seq_len_shape,max_seq_len)
            prefill_graphs.append((tensor_args, graph))
            assert batch_size == self._extend_seq_len_interval*len(prefill_graphs), f"batch size {batch_size} does not match the number of graphs {len(self._decode_graphs)}"

        return prefill_graphs


    
    def _run_sdpa_forward_extend(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
    ):
        """Run the extend forward by using torch native sdpa op.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_heads, head_size]
            v_cache: [max_total_num_tokens, num_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            extend_prefix_lens: [num_seqs]
            extend_seq_lens: [num_seqs]
            scaling: float or None
            enable_gqa: bool
            causal: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """
        assert seq_lens.shape[0] == extend_prefix_lens.shape[0]
        assert seq_lens.shape[0] == extend_seq_lens.shape[0]

        

        start_time = time.perf_counter()

        # B = batch_size
        B = seq_lens.shape[0]

        assert query.shape[0] == extend_seq_lens.sum(), (
            f"query.shape[0] = {query.shape[0]}, but sum(extend_seq_lens) = {extend_seq_lens.sum()}"
        )


        # how many tokens can store in KV cache
        max_seq_len = k_cache.shape[0]

        # 1) Reshape the multi-token query to [B, max_new_tokens, H, D] in a padded fashion
        H = query.shape[1]
        D = query.shape[2]
        max_new_tokens = extend_seq_lens.max()
        # find the first prefill graph whose sequence length is longer than max_new_tokens

        for i, (tensor_args, graph) in enumerate(self._prefill_graphs):
            if i * self._extend_seq_len_interval >= max_new_tokens:
                break
        tensor_args, graph = self._prefill_graphs[i]
        pad_size = i * self._extend_seq_len_interval

        # TODO: maybe using ragged tensor?
        padded_query = query.new_zeros((B, pad_size, H, D))

        # Fill in each sequence's slice
        offset = 0
        for i in range(B):
            length_i = extend_seq_lens[i].item()
            padded_query[i, :length_i, :, :] = query[offset : offset + length_i, :, :]
            offset += length_i

        # [B, H, max_new_tokens, D]
        padded_query = padded_query.movedim(2, 1)



        # query contains num_tokens queries batched togather
        q_gpu = padded_query

        # heads, tokens, head size
        # The tokens of queries are indexed by req_to_token
        s, h, d = k_cache.shape
        # Block Size of Paged Cache, 1 since only one token per block
        b = 1

        # 3) Reshape k_cache, v_cache into container shapes for “paged” attention
        # container: num_blocks, num heads, tokens_per_block, dim
        # TODO: permute for correctness
        container_k_gpu = k_cache.view(s,h,b,d)
        print('cache shape: ',container_k_gpu.shape)
        container_v_gpu = v_cache.view(s,h,b,d)

        # 4) Build the page table
        # only want prefix + the newly added tokens for each sequence
        # Then pad it to the maximum across the batch
        max_ctx_len = (extend_prefix_lens + extend_seq_lens).max().item()
        list_req_tokens = []
        for i in range(B):
            total_len = (extend_prefix_lens[i] + extend_seq_lens[i]).item()
            row_i = req_to_token[req_pool_indices[i], :total_len]
            # Pad up to max_ctx_len
            padded_indices = row_i.new_zeros(max_ctx_len)
            padded_indices[:total_len] = row_i
            list_req_tokens.append(padded_indices)
        
        # Now stack into a single [B, max_ctx_len]
        per_req_tokens = torch.stack(list_req_tokens, dim=0)

        # reshape to [B, 1, max_ctx_len, 1]
        page_table_k_gpu = per_req_tokens.view(per_req_tokens.shape[0],1,per_req_tokens.shape[1],1)
        print("paged table k shape: ",page_table_k_gpu.shape)
        page_table_v_gpu = per_req_tokens.view(per_req_tokens.shape[0],1,per_req_tokens.shape[1],1)
        print("page table v shape: ",page_table_v_gpu.shape)

        # 5) Sequence lengths
        seq_lens_kv = (extend_prefix_lens + extend_seq_lens).view(B, 1, 1, 1)
        seq_lens_q = extend_seq_lens.view(B, 1, 1, 1)


        # 7) Set output tensor
        # CuDNN output will also be [B, H, max_new_tokens, D]
        # eventually flatten it back to [sum_of_new_tokens_across_batch, H, D]
        
        #output = output.view(*padded_query.shape)
        B_out, H_out, S_out, D_out = padded_query.shape
        output = output.new_zeros((B_out, H_out, S_out, D_out))

        build_graph_time = time.perf_counter()

        variable_pack = {
            tensor_args[self._ArgMapKeys.q]: q_gpu,
            tensor_args[self._ArgMapKeys.k_container]: container_k_gpu,
            tensor_args[self._ArgMapKeys.v_container]: container_v_gpu,
            tensor_args[self._ArgMapKeys.k_page_table]: page_table_k_gpu,
            tensor_args[self._ArgMapKeys.v_page_table]: page_table_v_gpu,  
            tensor_args[self._ArgMapKeys.seq_len_q_tensor_info]: seq_lens_q,
            tensor_args[self._ArgMapKeys.seq_len_kv_tensor_info]: seq_lens_kv,
            tensor_args[self._ArgMapKeys.o]: output,
        }


        workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
        graph.execute(variable_pack, workspace)
        print(output.shape)

        # 8) Reshape the output back to [sum_of_new_tokens_across_batch, H, D]
        final_out = []
        offset = 0
        for i in range(B):
            length_i = extend_seq_lens[i].item()
            seq_out = output[i, :, :length_i, :] 
            # permute => [length_i, H, D]
            seq_out = seq_out.movedim(0, 1)
            final_out.append(seq_out)
        final_output = torch.cat(final_out, dim=0)
        end_time = time.perf_counter()

        print(f"Graph Construction Time: {build_graph_time-start_time}")
        print(f"Forward Time: {end_time-build_graph_time}")

        return final_output

      

    def _run_sdpa_forward_decode(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=True,
    ):
        """Run the decode forward by using torch native sdpa op.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_heads, head_size]
            v_cache: [max_total_num_tokens, num_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            scaling: float or None
            enable_gqa: bool
            causal: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """
        assert query.shape[0] == seq_lens.shape[0], "batch size must be the same"

        tensor_key_map,cudnn_decode_graph = self._decode_graphs[query.shape[0]-1]
        # Convert into CuDNN Query format (B, H, S, D)
        # where B is number of queries and S is sequence per query (1 in decoding)
        # [num_tokens, num_heads, head_size] -> [num_token, num_heads, 1,  head_size]
        query = query.unsqueeze(1).movedim(1,2)


        # heads, tokens, head size
        # The tokens of queries are indexed by req_to_token
        s, h, d = k_cache.shape
        # Block Size of Paged Cache, 1 since only one token per block
        b = 1

        # Radix Attention use KVCache of Block Size 1

        # get the request id of each query up to t
        # per_req_tokens = req_to_token[req_pool_indices, :seq_len_kv]

        # get the token location in kvcache, only up to seq_len_kv is valid
        # cudnn required shape: (num_block, 1, ceil(s/num_block), 1)
        per_req_tokens = req_to_token[req_pool_indices, :]

        # get the kv cache with request id
        # container: num_blocks, num heads, tokens_per_block, dim
        container_k_gpu = k_cache.view(s,h,b,d)
        container_v_gpu = v_cache.view(s,h,b,d)

        page_table_k_gpu = per_req_tokens.view(per_req_tokens.shape[0],1,per_req_tokens.shape[1],1)
        page_table_v_gpu = per_req_tokens.view(per_req_tokens.shape[0],1,per_req_tokens.shape[1],1)
        
        seq_lens_kv = seq_lens.view(seq_lens.shape[0], 1, 1, 1)
        seq_lens_q = torch.ones_like(seq_lens_kv)


        output = output.view(*query.shape)
        variant_pack = {
            tensor_key_map[self._ArgMapKeys.q]: query,
            tensor_key_map[self._ArgMapKeys.k_container]: container_k_gpu,
            tensor_key_map[self._ArgMapKeys.v_container]: container_v_gpu,
            tensor_key_map[self._ArgMapKeys.k_page_table]: page_table_k_gpu,
            tensor_key_map[self._ArgMapKeys.v_page_table]: page_table_v_gpu,  
            tensor_key_map[self._ArgMapKeys.seq_len_q_tensor_info]: seq_lens_q,
            tensor_key_map[self._ArgMapKeys.seq_len_kv_tensor_info]: seq_lens_kv,
            tensor_key_map[self._ArgMapKeys.o]: output,
        }

    
        workspace = torch.empty(cudnn_decode_graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
        cudnn_decode_graph.execute(variant_pack, workspace)

        return output

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        self._run_sdpa_forward_extend(
            q_,
            o_,
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.extend_prefix_lens,
            forward_batch.extend_seq_lens,
            scaling=layer.scaling,
            enable_gqa=use_gqa,
            causal=not layer.is_cross_attention,
        )
        return o

    def forward_decode(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        # During torch.compile, there is a bug in rotary_emb that causes the
        # output value to have a 3D tensor shape. This reshapes the output correctly.
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        self._run_sdpa_forward_decode(
            q_,
            o_,
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            scaling=layer.scaling,
            enable_gqa=use_gqa,
            causal=False,
        )

        return o
