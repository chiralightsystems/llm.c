# Integrated decoder prerequisite patches

Search tags: History; `wm:llmc`; `wm:decode`; `wm:provenance`.

These are byte-for-byte records of the parent worldmodel cached-decoder patches.
Their changes are already present in this branch. Do not apply them again.

Original application order on llm.c `247f4384` was attention workspace,
validation timing, small-vocabulary classifier, LayerNorm decode guard,
Linux C++20, and inference NoPE. The classifier change is included here through
the original `0e09f14` commit; the other five were applied after merging the
existing fork histories. Full origins and input hashes are in
[`../../provenance.json`](../../provenance.json).
