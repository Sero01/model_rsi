"""Write an F32 GGUF of Qwen3.5-0.8B's text model from the verified safetensors, for llama.cpp's qwen35 architecture.

    python -I phase1/make_gguf.py <model dir> <model.safetensors> <out.gguf>

Our own writer: tensor names, transforms and metadata follow llama.cpp b12099 conversion/qwen.py and
gguf-py/gguf/tensor_mapping.py, read but not run. F32 represents every BF16 weight exactly. Vision and MTP tensors are
dropped, as vLLM does not use them for text generation. The V-head reorder in the converter is a no-op here because
linear_num_key_heads == linear_num_value_heads.
"""
import json
import os
import re
import sys

import gguf
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import st_numpy as S  # noqa: E402

ARCH = "qwen35"
TOKENIZER_PRE = "qwen35"
PREFIX = "model.language_model."
EOS_TOKEN, PAD_TOKEN = "<|im_end|>", "<|endoftext|>"
MROPE_SECTIONS_LEN = 4

GLOBAL_NAMES = {"embed_tokens.weight": "token_embd.weight", "norm.weight": "output_norm.weight"}
LAYER_NAMES = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "post_attention_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "linear_attn.in_proj_qkv.weight": "attn_qkv.weight",
    "linear_attn.in_proj_z.weight": "attn_gate.weight",
    "linear_attn.in_proj_a.weight": "ssm_alpha.weight",
    "linear_attn.in_proj_b.weight": "ssm_beta.weight",
    "linear_attn.out_proj.weight": "ssm_out.weight",
    "linear_attn.conv1d.weight": "ssm_conv1d.weight",
    "linear_attn.norm.weight": "ssm_norm.weight",
    "linear_attn.dt_bias": "ssm_dt.bias",
    "linear_attn.A_log": "ssm_a",
}


def gguf_name(hf_name):
    """The GGUF name of a text tensor, or None for tensors the text model does not use."""
    if not hf_name.startswith(PREFIX):
        return None
    rest = hf_name[len(PREFIX):]
    if rest in GLOBAL_NAMES:
        return GLOBAL_NAMES[rest]
    match = re.fullmatch(r"layers\.(\d+)\.(.+)", rest)
    if match is None or match.group(2) not in LAYER_NAMES:
        raise SystemExit(f"unmapped tensor: {hf_name}")
    return f"blk.{match.group(1)}.{LAYER_NAMES[match.group(2)]}"


def transform(hf_name, values):
    """The value changes in the converter's Qwen3NextModel.modify_tensors."""
    if hf_name.endswith(".A_log"):
        return -np.exp(values)
    if ".conv1d." in hf_name:
        return np.squeeze(values)
    if hf_name.endswith("norm.weight") and not hf_name.endswith("linear_attn.norm.weight"):
        return values + 1.0
    return values


def to_f32(dtype, array):
    if dtype == "BF16":
        return S.bf16_to_f32(array)
    if dtype == "F32":
        return array
    raise SystemExit(f"unexpected dtype {dtype}")


def read_tensor(f, data_start, info):
    a, b = info["data_offsets"]
    f.seek(data_start + a)
    return np.frombuffer(f.read(b - a), dtype=S.DT[info["dtype"]]).reshape(info["shape"])


def text_config(model_dir):
    config = json.load(open(os.path.join(model_dir, "config.json")))
    return config["text_config"]


def add_vocab(writer, model_dir, vocab_size):
    vocab = json.load(open(os.path.join(model_dir, "vocab.json")))
    added = json.load(open(os.path.join(model_dir, "tokenizer_config.json")))["added_tokens_decoder"]
    tokens = [f"[PAD{i}]" for i in range(vocab_size)]
    types = [gguf.TokenType.UNUSED] * vocab_size
    for token, i in vocab.items():
        tokens[i], types[i] = token, gguf.TokenType.NORMAL
    for i, spec in added.items():
        i = int(i)
        tokens[i] = spec["content"]
        types[i] = gguf.TokenType.CONTROL if spec["special"] else gguf.TokenType.USER_DEFINED
    with open(os.path.join(model_dir, "merges.txt"), encoding="utf-8") as f:
        merges = [line.rstrip("\n") for line in f if line.strip() and not line.startswith("#version")]

    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre(TOKENIZER_PRE)
    writer.add_token_list(tokens)
    writer.add_token_types(types)
    writer.add_token_merges(merges)
    writer.add_eos_token_id(tokens.index(EOS_TOKEN))
    writer.add_pad_token_id(tokens.index(PAD_TOKEN))
    writer.add_add_bos_token(False)
    with open(os.path.join(model_dir, "chat_template.jinja"), encoding="utf-8") as f:
        writer.add_chat_template(f.read())


def add_hparams(writer, c):
    rope = c["rope_parameters"]
    sections = list(rope["mrope_section"]) + [0] * (MROPE_SECTIONS_LEN - len(rope["mrope_section"]))
    a = ARCH
    writer.add_name("Qwen3.5-0.8B")
    writer.add_uint32(f"{a}.block_count", c["num_hidden_layers"])
    writer.add_uint32(f"{a}.context_length", c["max_position_embeddings"])
    writer.add_uint32(f"{a}.embedding_length", c["hidden_size"])
    writer.add_uint32(f"{a}.feed_forward_length", c["intermediate_size"])
    writer.add_uint32(f"{a}.attention.head_count", c["num_attention_heads"])
    writer.add_uint32(f"{a}.attention.head_count_kv", c["num_key_value_heads"])
    writer.add_uint32(f"{a}.attention.key_length", c["head_dim"])
    writer.add_uint32(f"{a}.attention.value_length", c["head_dim"])
    writer.add_float32(f"{a}.attention.layer_norm_rms_epsilon", c["rms_norm_eps"])
    writer.add_array(f"{a}.attention.recurrent_layers", [t == "linear_attention" for t in c["layer_types"]])
    writer.add_uint32(f"{a}.full_attention_interval", c["full_attention_interval"])
    writer.add_float32(f"{a}.rope.freq_base", rope["rope_theta"])
    writer.add_uint32(f"{a}.rope.dimension_count", int(c["head_dim"] * rope["partial_rotary_factor"]))
    writer.add_array(f"{a}.rope.dimension_sections", sections)
    writer.add_uint32(f"{a}.vocab_size", c["vocab_size"])
    writer.add_uint32(f"{a}.ssm.conv_kernel", c["linear_conv_kernel_dim"])
    writer.add_uint32(f"{a}.ssm.state_size", c["linear_key_head_dim"])
    writer.add_uint32(f"{a}.ssm.group_count", c["linear_num_key_heads"])
    writer.add_uint32(f"{a}.ssm.time_step_rank", c["linear_num_value_heads"])
    writer.add_uint32(f"{a}.ssm.inner_size", c["linear_value_head_dim"] * c["linear_num_value_heads"])
    writer.add_file_type(gguf.LlamaFileType.ALL_F32)


def main(model_dir, weights, out):
    c = text_config(model_dir)
    if c["linear_num_key_heads"] != c["linear_num_value_heads"]:
        raise SystemExit("this writer does not implement the V-head reorder")
    header, data_start = S.read_header(weights)
    header.pop("__metadata__", None)
    names = sorted(n for n in header if gguf_name(n) is not None)

    writer = gguf.GGUFWriter(out, ARCH)
    add_hparams(writer, c)
    add_vocab(writer, model_dir, c["vocab_size"])
    for hf_name in names:
        shape = [d for d in header[hf_name]["shape"] if d != 1] if ".conv1d." in hf_name else header[hf_name]["shape"]
        nbytes = int(np.prod(shape)) * np.dtype(np.float32).itemsize
        writer.add_tensor_info(gguf_name(hf_name), shape, np.dtype(np.float32), nbytes, raw_dtype=gguf.GGMLQuantizationType.F32)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    with open(weights, "rb") as f:
        for hf_name in names:
            info = header[hf_name]
            values = transform(hf_name, to_f32(info["dtype"], read_tensor(f, data_start, info)))
            writer.write_tensor_data(np.ascontiguousarray(values, dtype=np.float32))
    writer.close()
    print(f"{len(names)} tensors -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
