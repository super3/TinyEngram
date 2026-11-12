import torch
import torch.nn as nn
from transformers import Qwen3ForCausalLM, Qwen3Model, AutoConfig, Qwen3Config
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from typing import Optional
from loguru import logger
from engram import Engram, EngramConfig

SKIP_ENGRAM = True
_WARNED_CACHED_DECODE = False


def set_skip_engram(value: bool):
    global SKIP_ENGRAM
    SKIP_ENGRAM = value


class EngramQwenModel(Qwen3Model):
    """
    A Qwen3Model variant that injects Engram layers at specified indices,
    without reimplementing Qwen3's forward (keeps rotary/position embeddings intact).
    """

    def __init__(self, config, engram_cfg: EngramConfig, engram_cls=None):
        super().__init__(config)

        self.engram_cfg = engram_cfg
        engram_cls = Engram if engram_cls is None else engram_cls

        self.engram_layers = nn.ModuleDict()
        for layer_id in self.engram_cfg.layer_ids:
            self.engram_layers[str(layer_id)] = engram_cls(
                layer_id=layer_id,
                config=self.engram_cfg,
                backbone_hidden_size=config.hidden_size,
                backbone_hc_mult=1,
            )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = (
                    create_sliding_window_causal_mask(**mask_kwargs)
                )
        else:
            causal_mask_mapping = attention_mask

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Cached decoding: Engram modules that support it keep their n-gram and conv
        # history in a dict stored on the KV cache object, reset at the first chunk.
        is_continuation = past_key_values is not None and int(cache_position[0]) > 0
        engram_store = None
        if past_key_values is not None:
            engram_store = getattr(past_key_values, "_engram_state", None)
            if engram_store is None or not is_continuation:
                engram_store = {}
                past_key_values._engram_state = engram_store
        token_mask = (
            attention_mask
            if isinstance(attention_mask, torch.Tensor) and attention_mask.dim() == 2
            else None
        )

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            # Inject Engram if configured for this layer
            if str(i) in self.engram_layers:
                module = self.engram_layers[str(i)]
                if (
                    not SKIP_ENGRAM
                    and input_ids is not None
                    and getattr(module, "enabled", True)
                ):
                    hs_expanded = hidden_states.unsqueeze(2)  # [B, L, 1, D]
                    if getattr(module, "supports_incremental", False):
                        state = (
                            engram_store.setdefault(str(i), {})
                            if engram_store is not None
                            else None
                        )
                        engram_out = module(
                            hidden_states=hs_expanded,
                            input_ids=input_ids,
                            state=state,
                            token_mask=token_mask,
                        )
                    else:
                        if is_continuation:
                            _warn_cached_decode()
                        engram_out = module(
                            hidden_states=hs_expanded, input_ids=input_ids
                        )
                    hidden_states = hidden_states + engram_out.squeeze(2).to(
                        hidden_states.dtype
                    )

            # wrapper's __getattr__ proxies attention_type if wrapped
            attn_mask = causal_mask_mapping[decoder_layer.attention_type]

            layer_kwargs = {
                "attention_mask": attn_mask,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "cache_position": cache_position,
                "position_embeddings": position_embeddings,
            }

            hidden_states = decoder_layer(hidden_states, **layer_kwargs, **kwargs)

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


def _warn_cached_decode():
    global _WARNED_CACHED_DECODE
    if not _WARNED_CACHED_DECODE:
        logger.warning(
            "Engram is running under cached decoding: the base Engram module only sees "
            "the new token, so its n-grams use padding instead of the real history and "
            "the output differs from a full forward pass. Use "
            "domain_tables.model.DomainEngramQwenForCausalLM for correct decoding."
        )
        _WARNED_CACHED_DECODE = True


class EngramQwenForCausalLM(Qwen3ForCausalLM):
    # Subclasses can set this to swap the Engram module class. Prefer that over passing
    # engram_cls to from_pretrained: transformers forwards unknown from_pretrained kwargs
    # into GenerationConfig (that is how engram_cfg ends up in generation_config.json),
    # and a class object there is not JSON serializable.
    engram_cls = None

    def __init__(self, config, *, engram_cfg: EngramConfig, engram_cls=None):
        engram_cls = engram_cls or type(self).engram_cls
        # Initialize the original Qwen structure so pretrained weights can be loaded
        super().__init__(config)
        # Replace the backbone with our Engram-aware model
        # The structure (layers, norm, etc.) matches, so load_state_dict works for Qwen weights
        self.model = EngramQwenModel(config, engram_cfg=engram_cfg, engram_cls=engram_cls)
        self.tie_weights()


if __name__ == "__main__":
    print("🚀 Initializing Engram-Qwen Integration...")

    # 1. Load Qwen config (using standard Qwen3-0.5B for demo dimensions)
    model_path = "/nasdata/model/Qwen/Qwen3-0___6B"
    # config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config = Qwen3Config(
        vocab_size=100,
        hidden_size=128,
        num_hidden_layers=4,
        num_key_value_heads=4,
        num_attention_heads=8,
        intermediate_size=256,
    )

    # 2. Instantiate Modified Model
    model = EngramQwenForCausalLM(
        config,
        engram_cfg=EngramConfig(
            tokenizer_name_or_path=model_path,
            engram_vocab_size=[10000, 10000],
            layer_ids=[1, 3],
        ),
    )

    # 3. Test Forward
    input_ids = torch.randint(0, config.vocab_size, (1, 20))
    outputs = model(input_ids)

    print("✅ Forward Complete!")
    print(f"{input_ids.shape=}")
    print(f"{outputs.logits.shape=}")
