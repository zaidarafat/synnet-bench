# Chapter companion scripts

One script per chapter (ch01–ch23; Chapter 24 has no experiment). Each reproduces
every number, table and figure in its chapter.

**Paths.** The scripts were written in the authoring environment and carry absolute
paths at the top (e.g. `DATA = Path("/home/claude/data/nslkdd/KDDTrain+.txt")`) and,
for a few later chapters, `exec(...)` of an earlier chapter's script to reuse its
generators (e.g. Chapter 22 reuses Chapter 13). Before running, adjust the `DATA`
path to your local NSL-KDD files (see ../data/README.md) and, where a script execs
another, keep the `chapters/` scripts together and fix the referenced path.

**Reproducibility.** Every script fixes all seeds. Long experiments cache per-job
results under `out/runs/` and resume if interrupted. The `synnet/` library is the
cleaned-up, unified version of the models these scripts use; prefer it for new work.
