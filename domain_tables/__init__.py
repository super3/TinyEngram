"""Domain n-gram tables on top of TinyEngram.

Extensions to the base ``Engram`` module for the domain-table experiments:

* ``PhraseList``: exact allowlist of registered n-grams (phrase selection).
* ``BloomFilter``: compact, serving-time approximation of a phrase list.
* ``DomainEngram``: an ``Engram`` subclass with selectivity masking, occupancy
  tracking, frozen-table controls, telemetry and correct cached decoding.
* ``save_tables`` / ``load_tables``: table-only checkpoints (no base weights).
* ``DomainEngramQwenForCausalLM`` (in ``domain_tables.model``): EngramQwen with
  DomainEngram layers; imported lazily because it pulls in the Qwen3 model code.
"""

from .ngram_keys import ngram_keys, tag_keys
from .bloom import BloomFilter
from .phrase_list import PhraseList
from .table import DomainTableConfig, DomainEngram
from .io import save_tables, load_tables, read_table_config

__all__ = [
    "ngram_keys",
    "tag_keys",
    "BloomFilter",
    "PhraseList",
    "DomainTableConfig",
    "DomainEngram",
    "save_tables",
    "load_tables",
    "read_table_config",
]
