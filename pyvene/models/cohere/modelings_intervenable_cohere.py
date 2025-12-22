"""
Cohere decoder-only causal language model (HF transformers) support for pyvene.

Maps abstract intervention anchor points to concrete Cohere model modules,
and defines how intervention dimensions are inferred from CohereConfig.
"""

from ..constants import *

############################################
# Base Cohere (decoder-only, no LM head)
############################################

cohere_type_to_module_mapping = {
    # ===== Block-level =====
    "block_input": ("layers[%s]", CONST_INPUT_HOOK),
    "block_output": ("layers[%s]", CONST_OUTPUT_HOOK),

    # ===== LayerNorm =====
    "attention_input": ("layers[%s].input_layernorm", CONST_OUTPUT_HOOK),

    # ===== MLP =====
    "mlp_input": ("layers[%s].mlp", CONST_INPUT_HOOK),
    "mlp_output": ("layers[%s].mlp", CONST_OUTPUT_HOOK),
    "mlp_activation": (
        "layers[%s].mlp.gate_proj",
        CONST_OUTPUT_HOOK,
    ),

    # ===== Attention (combined output) =====
    "attention_output": (
        "layers[%s].self_attn.o_proj",
        CONST_INPUT_HOOK,
    ),

    # ===== Attention projections =====
    "query_output": (
        "layers[%s].self_attn.q_proj",
        CONST_OUTPUT_HOOK,
    ),
    "key_output": (
        "layers[%s].self_attn.k_proj",
        CONST_OUTPUT_HOOK,
    ),
    "value_output": (
        "layers[%s].self_attn.v_proj",
        CONST_OUTPUT_HOOK,
    ),

    # ===== Head-level projections =====
    "head_query_output": (
        "layers[%s].self_attn.q_proj",
        CONST_OUTPUT_HOOK,
        (split_head_and_permute, "num_attention_heads"),
    ),
    "head_key_output": (
        "layers[%s].self_attn.k_proj",
        CONST_OUTPUT_HOOK,
        (split_head_and_permute, "num_key_value_heads"),
    ),
    "head_value_output": (
        "layers[%s].self_attn.v_proj",
        CONST_OUTPUT_HOOK,
        (split_head_and_permute, "num_key_value_heads"),
    ),
    "head_attention_value_output": (
        "layers[%s].self_attn.o_proj",
        CONST_INPUT_HOOK,
        (split_head_and_permute, "num_attention_heads"),
    ),
}

############################################
# Dimension inference
############################################

cohere_type_to_dimension_mapping = {
    # Core sizes
    "hidden_size": ("hidden_size",),
    "num_attention_heads": ("num_attention_heads",),
    "num_key_value_heads": ("num_key_value_heads",),
    "head_dim": ("head_dim", "hidden_size/num_attention_heads"),

    # Block
    "block_input": ("hidden_size",),
    "block_output": ("hidden_size",),

    # MLP
    "mlp_input": ("hidden_size",),
    "mlp_output": ("hidden_size",),
    "mlp_activation": ("intermediate_size",),

    # Attention
    "attention_input": ("hidden_size",),
    "attention_output": ("hidden_size",),

    # Projections
    "query_output": ("hidden_size",),
    "key_output": ("hidden_size",),
    "value_output": ("hidden_size",),

    # Head-level
    "head_query_output": ("head_dim",),
    "head_key_output": ("head_dim",),
    "head_value_output": ("head_dim",),
}

############################################
# CohereModel (base)
############################################

def create_cohere(
    name="CohereForAI/c4ai-command-r-v01",
    cache_dir=None,
):
    """
    Creates CohereModel (no LM head), config, tokenizer
    """
    from transformers import AutoTokenizer, CohereConfig, CohereModel

    config = CohereConfig.from_pretrained(name)
    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = "eager"

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = CohereModel.from_pretrained(
        name,
        config=config,
        cache_dir=cache_dir,
    )
    print("Loaded CohereModel")
    return config, tokenizer, model

############################################
# CohereForCausalLM
############################################

cohere_lm_type_to_module_mapping = {}
for k, v in cohere_type_to_module_mapping.items():
    cohere_lm_type_to_module_mapping[k] = (f"model.{v[0]}",) + v[1:]

cohere_lm_type_to_dimension_mapping = cohere_type_to_dimension_mapping


def create_cohere_lm(
    name="CohereForAI/c4ai-command-r-v01",
    config=None,
    cache_dir=None,
):
    """
    Creates CohereForCausalLM, config, tokenizer
    """
    from transformers import AutoTokenizer, CohereConfig, CohereForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(name)

    if config is None:
        config = CohereConfig.from_pretrained(name)
        if hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"
        model = CohereForCausalLM.from_pretrained(
            name,
            config=config,
            cache_dir=cache_dir,
        )
    else:
        if hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"
        model = CohereForCausalLM(config=config)

    print("Loaded CohereForCausalLM")
    return config, tokenizer, model
