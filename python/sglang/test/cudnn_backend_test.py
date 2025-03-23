from dataclasses import dataclass
#from sglang.srt.layers.attention import cudnn_backend
import torch
import logging
import math
import cudnn

@dataclass
class InputParameters:
    num_token = 10
    num_heads = 4
    head_size = 128
    max_total_num_tokens = 300
    max_num_reqs = 100
    max_context_lenght = 300
    num_seqs = 10

class CuDNNBackend():
    def __init__(self):
        super().__init__()
        self.forward_metadata = None

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
        pass

      

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
        causal=False,
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

        logging.info("Running decode")
        print("output shape: ",output.shape)
        assert query.shape[0] == seq_lens.shape[0], "batch size must be the same"

        max_seq_len = k_cache.shape[0]
        # Convert into CuDNN Query format (B, H, S, D)
        # where B is number of queries and S is sequence per query (1 in decoding)
        # [num_tokens, num_heads, head_size] -> [num_token, num_heads, 1,  head_size]
        query = query.unsqueeze(1).movedim(1,2)
        print(query.shape)
        print(query.device)

        # heads, tokens, head size
        # The tokens of queries are indexed by req_to_token
        s, h, d = k_cache.shape
        # Block Size of Paged Cache, 1 since only one token per block
        b = 1

        # Radix Attention use KVCache of Block Size 1

        # TODO: determine data type
        graph = cudnn.pygraph(
            io_data_type=cudnn.data_type.HALF,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
        )
        print(graph)

        # query contains num_tokens queries batched togather
        q_gpu = query
        q = graph.tensor_like(q_gpu)


        # get the request id of each query up to t
        # per_req_tokens = req_to_token[req_pool_indices, :seq_len_kv]

        # get the token location in kvcache, only up to seq_len_kv is valid
        # cudnn required shape: (num_block, 1, ceil(s/num_block), 1)
        print("req index shape: ",req_pool_indices.shape,"req to token shape: ",req_to_token.shape)
        per_req_tokens = req_to_token[req_pool_indices, :]
        print("per req token shape: ",per_req_tokens.shape)

        # get the kv cache with request id
        # container: num_blocks, num heads, tokens_per_block, dim
        # TODO: permute for correctness
        container_k_gpu = k_cache.view(s,h,b,d)
        print('cache shape: ',container_k_gpu.shape)
        container_v_gpu = v_cache.view(s,h,b,d)


        container_k = graph.tensor_like(container_k_gpu)
        container_v = graph.tensor_like(container_v_gpu)


        page_table_k_gpu = per_req_tokens.view(per_req_tokens.shape[0],1,per_req_tokens.shape[1],1)
        print("paged table k shape: ",page_table_k_gpu.shape)
        page_table_v_gpu = per_req_tokens.view(per_req_tokens.shape[0],1,per_req_tokens.shape[1],1)
        print("page table v shape: ",page_table_v_gpu.shape)
        page_table_k = graph.tensor_like(page_table_k_gpu)
        page_table_v = graph.tensor_like(page_table_v_gpu)

        seq_lens_kv = seq_lens.view(seq_lens.shape[0], 1, 1, 1)

        seq_lens_q = torch.ones_like(seq_lens_kv)

        seq_len_q_tensor_info = graph.tensor_like(seq_lens_q)
        seq_len_kv_tensor_info = graph.tensor_like(seq_lens_kv)

        o, _ = graph.sdpa(
            name="sdpa",
            q=q,
            k=container_k,  # Container K: non contiguous container with K blocks
            v=container_v,  # Container V: non contiguous container with V blocks
            is_inference=True,
            attn_scale=scaling,
            use_causal_mask=True,
            use_padding_mask=True,
            seq_len_q=seq_len_q_tensor_info,
            seq_len_kv=seq_len_kv_tensor_info,
            paged_attention_k_table=page_table_k,  # Page Table K: Tensor containing offsets to the container with K blocks
            paged_attention_v_table=page_table_v,  # Page Table V: Tensor containing offsets to the container with V blocks
            paged_attention_max_seq_len_kv=max_seq_len,  # The maximum sequence length for K caches (this is optional, but recommended)
        )
        logging.info(graph)

        output = output.view(*query.shape)
        dims = output.shape
        strides = output.stride()
        print("output shape: ",output.shape)


        o.set_output(True).set_dim(dims).set_stride(strides)
        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
        graph.build_plans()

        variant_pack = {
            q: q_gpu,
            container_k: container_k_gpu,
            container_v: container_v_gpu,
            page_table_k: page_table_k_gpu,
            page_table_v: page_table_v_gpu,
            seq_len_q_tensor_info: seq_lens_q,
            seq_len_kv_tensor_info: seq_lens_kv,
            o: output,
        }


        workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
        graph.execute(variant_pack, workspace)
        return output



def test_correctness():
    input_parem = InputParameters()
    cudnn_bknd = CuDNNBackend()
    # TODO: dtype
    query = torch.randn([input_parem.num_token, input_parem.num_heads, input_parem.head_size]).half().cuda()
    output = torch.randn([input_parem.num_token, input_parem.num_heads, input_parem.head_size]).half().cuda()
    k_cache = torch.randn([input_parem.max_total_num_tokens, input_parem.num_heads, input_parem.head_size]).half().cuda()
    v_cache = torch.randn([input_parem.max_total_num_tokens, input_parem.num_heads, input_parem.head_size]).half().cuda()

    # the following are int tensors

    # the request index of inputs sequences in req_to_token
    req_pool_indices = torch.randint(low=0,high=input_parem.max_num_reqs,size=[input_parem.num_seqs],dtype=torch.int32).cuda()

    # req_to_token[request_index]: list of index of tokens in query and value for that request_index
    # sum(len(tokens_per_request)) = num_tokens in query
    req_to_token = torch.randint(low=0,high=input_parem.num_token,size=[input_parem.max_num_reqs, input_parem.max_context_lenght],dtype=torch.int32).cuda()

    seq_lens = torch.randint(low=0,high=input_parem.num_token,size=[input_parem.num_seqs]).cuda()
    # extend_prefix_lens = torch.randint(low=0,high=1,size=[input_parem.num_seqs],dtype=torch.int32)
    # extend_seq_lens = torch.randint(low=0,high=1,size=[input_parem.num_seqs],dtype=torch.int32)
    scaling = 1/math.sqrt(input_parem.head_size)

    logging.info("Start Extend")


    output = cudnn_bknd._run_sdpa_forward_decode(
        query=query,
        output=output,
        k_cache=k_cache,
        v_cache=v_cache,
        req_to_token=req_to_token,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        scaling=scaling
    )

    # TODO correctness
    print(output)

if __name__=='__main__':
    assert torch.cuda.is_available()
    assert (
        torch.cuda.get_device_capability()[0] >= 8
    ), f"SDPA operation is only supported on SM80 architecture (Ampere) or above, got {torch.cuda.get_device_capability()[0]}"

    # assert (
    #     cudnn.backend_version() >= 90500
    # ), f"SDPA operation is only supported cuDNN version 9.5.0 or above, got {cudnn.backend_version()}"
    test_correctness()



    