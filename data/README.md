# Datasets

The book develops its methods on **NSL-KDD**. Place the distribution files here:

    data/nslkdd/KDDTrain+.txt
    data/nslkdd/KDDTest+.txt

NSL-KDD is distributed by the Canadian Institute for Cybersecurity. See Chapter 18
and Appendix D for the dataset's provenance, known defects, and citation
(Tavallaee et al., 2009). The chapter scripts and `synnet.data.load_nslkdd` expect
the files at the path above; adjust the `DATA` path at the top of a chapter script
if you store them elsewhere.

Datasets are **not** redistributed in this repository.
