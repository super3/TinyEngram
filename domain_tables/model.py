"""EngramQwenForCausalLM with DomainEngram modules."""

from engram_qwen import EngramQwenForCausalLM

from .table import DomainEngram


class DomainEngramQwenForCausalLM(EngramQwenForCausalLM):
    """Use exactly like EngramQwenForCausalLM; every Engram layer is a DomainEngram.

    ``DomainEngramQwenForCausalLM.from_pretrained(base, engram_cfg=DomainTableConfig(...))``
    """

    engram_cls = DomainEngram
