PENDING = "[Reference to be completed on publication.]"
REFS = {
    "goodfellow2014": 'I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, "Generative adversarial nets," in *Advances in Neural Information Processing Systems 27 (NeurIPS)*, 2014, pp. 2672–2680.',
    "arjovsky_bottou2017": 'M. Arjovsky and L. Bottou, "Towards principled methods for training generative adversarial networks," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2017.',
    "kullback1951": 'S. Kullback and R. A. Leibler, "On information and sufficiency," *The Annals of Mathematical Statistics*, vol. 22, no. 1, pp. 79–86, 1951, doi: 10.1214/aoms/1177729694.',
    "cover2006": 'T. M. Cover and J. A. Thomas, *Elements of Information Theory*, 2nd ed. Hoboken, NJ: Wiley, 2006, doi: 10.1002/047174882X.',
    "lin1991": 'J. Lin, "Divergence measures based on the Shannon entropy," *IEEE Trans. Information Theory*, vol. 37, no. 1, pp. 145–151, 1991, doi: 10.1109/18.61115.',
    "ali_silvey1966": 'S. M. Ali and S. D. Silvey, "A general class of coefficients of divergence of one distribution from another," *J. Royal Statistical Society: Series B (Methodological)*, vol. 28, no. 1, pp. 131–142, 1966, doi: 10.1111/j.2517-6161.1966.tb00626.x.',
    "nowozin2016": 'S. Nowozin, B. Cseke, and R. Tomioka, "f-GAN: Training generative neural samplers using variational divergence minimization," in *Advances in Neural Information Processing Systems 29 (NeurIPS)*, 2016.',
    "muller1997": 'A. Müller, "Integral probability metrics and their generating classes of functions," *Advances in Applied Probability*, vol. 29, no. 2, pp. 429–443, 1997, doi: 10.2307/1428011.',
    "sriperumbudur2012": 'B. K. Sriperumbudur, K. Fukumizu, A. Gretton, B. Schölkopf, and G. R. G. Lanckriet, "On the empirical estimation of integral probability metrics," *Electronic Journal of Statistics*, vol. 6, pp. 1550–1599, 2012, doi: 10.1214/12-EJS722.',
    "gretton2012": 'A. Gretton, K. M. Borgwardt, M. J. Rasch, B. Schölkopf, and A. Smola, "A kernel two-sample test," *J. Machine Learning Research*, vol. 13, no. 25, pp. 723–773, 2012.',
    "villani2009": 'C. Villani, *Optimal Transport: Old and New*, Grundlehren der mathematischen Wissenschaften, vol. 338. Berlin: Springer, 2009, doi: 10.1007/978-3-540-71050-9.',
    "arjovsky2017": 'M. Arjovsky, S. Chintala, and L. Bottou, "Wasserstein generative adversarial networks," in *Proc. 34th Int. Conf. Machine Learning (ICML)*, PMLR vol. 70, 2017, pp. 214–223.',
    "gulrajani2017": 'I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. Courville, "Improved training of Wasserstein GANs," in *Advances in Neural Information Processing Systems 30 (NeurIPS)*, 2017.',
    "miyato2018": 'T. Miyato, T. Kataoka, M. Koyama, and Y. Yoshida, "Spectral normalization for generative adversarial networks," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2018.',
    "arafat_trafficgan": 'Z. Arafat and M. Ronkin, "TrafficGAN+: Mathematical modeling of a generative adversarial network for synthetic network traffic generation in intrusion detection systems." ' + PENDING,
    "sion1958": 'M. Sion, "On general minimax theorems," *Pacific J. Mathematics*, vol. 8, no. 1, pp. 171–176, 1958, doi: 10.2140/pjm.1958.8.171.',
    "nash1951": 'J. Nash, "Non-cooperative games," *Annals of Mathematics*, vol. 54, no. 2, pp. 286–295, 1951, doi: 10.2307/1969529.',
    "farnia2020": 'F. Farnia and A. Ozdaglar, "Do GANs always have Nash equilibria?," in *Proc. 37th Int. Conf. Machine Learning (ICML)*, PMLR vol. 119, 2020, pp. 3029–3039.',
    "nagarajan2017": 'V. Nagarajan and J. Z. Kolter, "Gradient descent GAN optimization is locally stable," in *Advances in Neural Information Processing Systems 30 (NeurIPS)*, 2017.',
    "mescheder2018": 'L. Mescheder, A. Geiger, and S. Nowozin, "Which training methods for GANs do actually converge?," in *Proc. 35th Int. Conf. Machine Learning (ICML)*, PMLR vol. 80, 2018, pp. 3481–3490.',
}
REFS.update({
    "salimans2016": 'T. Salimans, I. Goodfellow, W. Zaremba, V. Cheung, A. Radford, and X. Chen, "Improved techniques for training GANs," in *Advances in Neural Information Processing Systems 29 (NIPS 2016)*, 2016.',
    "metz2017": 'L. Metz, B. Poole, D. Pfau, and J. Sohl-Dickstein, "Unrolled generative adversarial networks," in *Proc. 5th Int. Conf. Learning Representations (ICLR)*, 2017.',
    "lin2018": 'Z. Lin, A. Khetan, G. Fanti, and S. Oh, "PacGAN: The power of two samples in generative adversarial networks," in *Advances in Neural Information Processing Systems 31 (NeurIPS 2018)*, 2018.',
    "srivastava2017": 'A. Srivastava, L. Valkov, C. Russell, M. U. Gutmann, and C. Sutton, "VEEGAN: Reducing mode collapse in GANs using implicit variational learning," in *Advances in Neural Information Processing Systems 30 (NIPS 2017)*, 2017.',
    "arafat2025": 'Z. Arafat, O. V. Yudina, and Z. A. Abdulazeez, "Generative adversarial networks in cyber security: Literature review," *Russian Technological Journal*, vol. 13, no. 5, pp. 7–24, 2025, doi: 10.32362/2500-316X-2025-13-5-7-24.',
    "biau2020": 'G. Biau, B. Cadre, M. Sangnier, and U. Tanielian, "Some theoretical properties of GANs," *The Annals of Statistics*, vol. 48, no. 3, pp. 1539–1566, 2020, doi: 10.1214/19-AOS1858.',
    "biau2021": 'G. Biau, M. Sangnier, and U. Tanielian, "Some theoretical insights into Wasserstein GANs," *J. Machine Learning Research*, vol. 22, no. 119, pp. 1–45, 2021.',
    "arora2017": 'S. Arora, R. Ge, Y. Liang, T. Ma, and Y. Zhang, "Generalization and equilibrium in generative adversarial nets (GANs)," in *Proc. 34th Int. Conf. Machine Learning (ICML)*, PMLR vol. 70, pp. 224–232, 2017.',
})
REFS.update({
    "lucic2018": 'M. Lucic, K. Kurach, M. Michalski, S. Gelly, and O. Bousquet, "Are GANs created equal? A large-scale study," in *Advances in Neural Information Processing Systems 31 (NeurIPS 2018)*, 2018.',
    "kurach2019": 'K. Kurach, M. Lučić, X. Zhai, M. Michalski, and S. Gelly, "A large-scale study on regularization and normalization in GANs," in *Proc. 36th Int. Conf. Machine Learning (ICML)*, PMLR vol. 97, pp. 3581–3590, 2019.',
    "mao2017": 'X. Mao, Q. Li, H. Xie, R. Y. K. Lau, Z. Wang, and S. P. Smolley, "Least squares generative adversarial networks," in *Proc. IEEE Int. Conf. Computer Vision (ICCV)*, 2017, pp. 2813–2821, doi: 10.1109/ICCV.2017.304.',
    "heusel2017": 'M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, "GANs trained by a two time-scale update rule converge to a local Nash equilibrium," in *Advances in Neural Information Processing Systems 30 (NIPS 2017)*, 2017.',
    "petzka2018": 'H. Petzka, A. Fischer, and D. Lukovnikov, "On the regularization of Wasserstein GANs," in *Proc. 6th Int. Conf. Learning Representations (ICLR)*, 2018.',
    "roth2017": 'K. Roth, A. Lucchi, S. Nowozin, and T. Hofmann, "Stabilizing training of generative adversarial networks through regularization," in *Advances in Neural Information Processing Systems 30 (NIPS 2017)*, 2017.',
    "zhang2019": 'H. Zhang, I. Goodfellow, D. Metaxas, and A. Odena, "Self-attention generative adversarial networks," in *Proc. 36th Int. Conf. Machine Learning (ICML)*, PMLR vol. 97, pp. 7354–7363, 2019.',
    "lim2017": 'J. H. Lim and J. C. Ye, "Geometric GAN," arXiv preprint arXiv:1705.02894, 2017. [Preprint]',
    "yin2022": 'Y. Yin, Z. Lin, M. Jin, G. Fanti, and V. Sekar, "Practical GAN-based synthetic IP header trace generation using NetShare," in *Proc. ACM SIGCOMM*, 2022, pp. 458–472, doi: 10.1145/3544216.3544251.',
})
REFS.update({
    "mirza2014": 'M. Mirza and S. Osindero, "Conditional generative adversarial nets," arXiv preprint arXiv:1411.1784, 2014. [Preprint]',
    "odena2017": 'A. Odena, C. Olah, and J. Shlens, "Conditional image synthesis with auxiliary classifier GANs," in *Proc. 34th Int. Conf. Machine Learning (ICML)*, PMLR vol. 70, pp. 2642–2651, 2017.',
    "chen2016": 'X. Chen, Y. Duan, R. Houthooft, J. Schulman, I. Sutskever, and P. Abbeel, "InfoGAN: Interpretable representation learning by information maximizing generative adversarial nets," in *Advances in Neural Information Processing Systems 29 (NIPS 2016)*, 2016.',
    "miyato_koyama2018": 'T. Miyato and M. Koyama, "cGANs with projection discriminator," in *Proc. 6th Int. Conf. Learning Representations (ICLR)*, 2018.',
    "gong2019": 'M. Gong, Y. Xu, C. Li, K. Zhang, and K. Batmanghelich, "Twin auxiliary classifiers GAN," in *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*, 2019.',
    "odena2016": 'A. Odena, "Semi-supervised learning with generative adversarial networks," arXiv preprint arXiv:1606.01583, 2016. [Preprint]',
    "dai2017": 'Z. Dai, Z. Yang, F. Yang, W. W. Cohen, and R. Salakhutdinov, "Good semi-supervised learning that requires a bad GAN," in *Advances in Neural Information Processing Systems 30 (NIPS 2017)*, 2017.',
    "aljubouri_cgan": 'R. M. Aljubouri, E. M. Abdulhussein, and Z. A. Abdulazeez, "Enhanced network intrusion detection through conditional GAN-based minority class augmentation," *Babylonian Journal of Artificial Intelligence*, vol. 2026, pp. 24–45, 2026, doi: 10.58496/BJAI/2026/011.',
})
REFS.update({
    "xu2019": 'L. Xu, M. Skoularidou, A. Cuesta-Infante, and K. Veeramachaneni, "Modeling tabular data using conditional GAN," in *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*, 2019.',
    "zhao2021": 'Z. Zhao, A. Kunar, R. Birke, and L. Y. Chen, "CTAB-GAN: Effective table data synthesizing," in *Proc. 13th Asian Conf. Machine Learning (ACML)*, PMLR vol. 157, pp. 97–112, 2021.',
    "kotelnikov2023": 'A. Kotelnikov, D. Baranchuk, I. Rubachev, and A. Babenko, "TabDDPM: Modelling tabular data with diffusion models," in *Proc. 40th Int. Conf. Machine Learning (ICML)*, PMLR vol. 202, pp. 17564–17579, 2023.',
    "stoian2024": 'M. C. Stoian, S. Dyrmishi, M. Cordy, T. Lukasiewicz, and E. Giunchiglia, "How realistic is your synthetic data? Constraining deep generative models for tabular data," in *Proc. 12th Int. Conf. Learning Representations (ICLR)*, 2024.',
    "grinsztajn2022": 'L. Grinsztajn, E. Oyallon, and G. Varoquaux, "Why do tree-based models still outperform deep learning on typical tabular data?," in *Advances in Neural Information Processing Systems 35 (NeurIPS 2022), Datasets and Benchmarks Track*, 2022.',
    "jang2017": 'E. Jang, S. Gu, and B. Poole, "Categorical reparameterization with Gumbel-softmax," in *Proc. 5th Int. Conf. Learning Representations (ICLR)*, 2017.',
    "sharafaldin2018": 'I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, "Toward generating a new intrusion detection dataset and intrusion traffic characterization," in *Proc. 4th Int. Conf. Information Systems Security and Privacy (ICISSP)*, 2018, pp. 108–116, doi: 10.5220/0006639801080116.',
    "lin2020": 'Z. Lin, A. Jain, C. Wang, G. Fanti, and V. Sekar, "Using GANs for sharing networked time series data: Challenges, initial promise, and open questions," in *Proc. ACM Internet Measurement Conf. (IMC)*, 2020, doi: 10.1145/3419394.3423643.',
    "ring2019": 'M. Ring, D. Schlör, D. Landes, and A. Hotho, "Flow-based network traffic generation using generative adversarial networks," *Computers & Security*, vol. 82, pp. 156–172, 2019, doi: 10.1016/j.cose.2018.12.012.',
    "pedregosa2011": 'F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *J. Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.',
})
REFS.update({
    "choi2017": 'E. Choi, S. Biswal, B. Malin, J. Duke, W. F. Stewart, and J. Sun, "Generating multi-label discrete patient records using generative adversarial networks," in *Proc. 2nd Machine Learning for Healthcare Conf. (MLHC)*, PMLR vol. 68, pp. 286–305, 2017.',
    "park2018": 'N. Park, M. Mohammadi, K. Gorde, S. Jajodia, H. Park, and Y. Kim, "Data synthesis based on generative adversarial networks," *Proc. VLDB Endowment*, vol. 11, no. 10, pp. 1071–1083, 2018, doi: 10.14778/3231751.3231757.',
    "kim2021": 'J. Kim, J. Jeon, J. Lee, J. Hyeong, and N. Park, "OCT-GAN: Neural ODE-based conditional tabular GANs," in *Proc. Web Conf. (WWW)*, 2021, pp. 1506–1515, doi: 10.1145/3442381.3449999.',
    "zhao2024": 'Z. Zhao, A. Kunar, R. Birke, H. Van der Scheer, and L. Y. Chen, "CTAB-GAN+: Enhancing tabular data synthesis," *Frontiers in Big Data*, vol. 6, 2023, doi: 10.3389/fdata.2023.1296508.',
    "patki2016": 'N. Patki, R. Wedge, and K. Veeramachaneni, "The synthetic data vault," in *Proc. IEEE Int. Conf. Data Science and Advanced Analytics (DSAA)*, 2016, pp. 399–410, doi: 10.1109/DSAA.2016.49.',
    "sdv_copulagan": 'DataCebo, "CopulaGANSynthesizer," SDV documentation. [Online]. Available: https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers/copulagansynthesizer (accessed Oct. 23, 2026).',
    "tavallaee2009": 'M. Tavallaee, E. Bagheri, W. Lu, and A. A. Ghorbani, "A detailed analysis of the KDD CUP 99 data set," in *Proc. IEEE Symp. Computational Intelligence for Security and Defense Applications (CISDA)*, 2009, pp. 1–6, doi: 10.1109/CISDA.2009.5356528.',
})
REFS.update({
    "arp2022": 'D. Arp, E. Quiring, F. Pendlebury, A. Warnecke, F. Pierazzi, C. Wressnegger, L. Cavallaro, and K. Rieck, "Dos and don\'ts of machine learning in computer security," in *Proc. 31st USENIX Security Symp.*, 2022, pp. 3971–3988.',
    "kapoor2023": 'S. Kapoor and A. Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science," *Patterns*, vol. 4, no. 9, art. 100804, 2023, doi: 10.1016/j.patter.2023.100804.',
    "liu2022": 'L. Liu, G. Engelen, T. Lynar, D. Essam, and W. Joosen, "Error prevalence in NIDS datasets: A case study on CIC-IDS-2017 and CSE-CIC-IDS-2018," in *Proc. IEEE Conf. Communications and Network Security (CNS)*, 2022, pp. 254–263, doi: 10.1109/CNS56114.2022.9947235.',
    "engelen2021": 'G. Engelen, V. Rimmer, and W. Joosen, "Troubleshooting an intrusion detection dataset: The CICIDS2017 case study," in *Proc. IEEE Security and Privacy Workshops (SPW)*, 2021, pp. 7–12, doi: 10.1109/SPW53761.2021.00009.',
    "chawla2002": 'N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic minority over-sampling technique," *J. Artificial Intelligence Research*, vol. 16, pp. 321–357, 2002, doi: 10.1613/jair.953.',
})
REFS.update({
    "esteban2017": 'C. Esteban, S. L. Hyland, and G. Rätsch, "Real-valued (medical) time series generation with recurrent conditional GANs," arXiv preprint arXiv:1706.02633, 2017. [Preprint]',
    "mogren2016": 'O. Mogren, "C-RNN-GAN: Continuous recurrent neural networks with adversarial training," arXiv preprint arXiv:1611.09904, 2016. [Preprint]',
    "yoon2019": 'J. Yoon, D. Jarrett, and M. van der Schaar, "Time-series generative adversarial networks," in *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*, 2019.',
    "xu2020": 'T. Xu, L. K. Wenliang, M. Munn, and B. Acciaio, "COT-GAN: Generating sequential data via causal optimal transport," in *Advances in Neural Information Processing Systems 33 (NeurIPS 2020)*, 2020, pp. 8798–8809.',
    "li2022": 'X. Li, V. Metsis, H. Wang, and A. H. H. Ngu, "TTS-GAN: A transformer-based time-series generative adversarial network," in *Artificial Intelligence in Medicine (AIME 2022)*, LNCS. Cham: Springer, 2022, pp. 133–143, doi: 10.1007/978-3-031-09342-5\\_13.',
})
REFS.update({
    "li2019flowgan": 'J. Li, L. Zhou, H. Li, L. Yan, and H. Zhu, "Dynamic traffic feature camouflaging via generative adversarial networks," in *Proc. IEEE Conf. Communications and Network Security (CNS)*, 2019, pp. 268–276, doi: 10.1109/CNS.2019.8802772.',
    "jiang2024": 'X. Jiang, S. Liu, A. Gember-Jacobson, A. N. Bhagoji, P. Schmitt, F. Bronzino, and N. Feamster, "NetDiffusion: Network data augmentation through protocol-constrained traffic generation," *Proc. ACM Measurement and Analysis of Computing Systems*, vol. 8, no. 1, pp. 1–32, 2024, doi: 10.1145/3639037.',
    "nukavarapu2022": 'S. K. Nukavarapu, M. Ayyat, and T. Nadeem, "MirageNet – Towards a GAN-based framework for synthetic network traffic generation," in *Proc. IEEE Global Communications Conf. (GLOBECOM)*, 2022.',
    "sivaroopan2023": 'N. Sivaroopan, D. Bandara, C. Madarasingha, G. Jourjon, A. Jayasumana, and K. Thilakarathna, "NetDiffus: Network traffic generation by diffusion models through time-series imaging," arXiv preprint arXiv:2310.04429, 2023. [Preprint]',
})
REFS.update({
    "kindji2025": 'G. C. N. Kindji, L. M. Rojas-Barahona, E. Fromont, and T. Urvoy, "Tabular data generation models: An in-depth survey and performance benchmarks with extensive tuning," *Neurocomputing*, vol. 658, art. 131655, 2025, doi: 10.1016/j.neucom.2025.131655.',
    "he2008": 'H. He, Y. Bai, E. A. Garcia, and S. Li, "ADASYN: Adaptive synthetic sampling approach for imbalanced learning," in *Proc. IEEE Int. Joint Conf. Neural Networks (IJCNN)*, 2008, pp. 1322–1328, doi: 10.1109/IJCNN.2008.4633969.',
    "han2005": 'H. Han, W.-Y. Wang, and B.-H. Mao, "Borderline-SMOTE: A new over-sampling method in imbalanced data sets learning," in *Advances in Intelligent Computing (ICIC 2005)*, LNCS 3644. Berlin: Springer, 2005, pp. 878–887, doi: 10.1007/11538059\\_91.',
    "kingma2014": 'D. P. Kingma and M. Welling, "Auto-encoding variational Bayes," in *Proc. 2nd Int. Conf. Learning Representations (ICLR)*, 2014.',
    "dinh2017": 'L. Dinh, J. Sohl-Dickstein, and S. Bengio, "Density estimation using Real NVP," in *Proc. 5th Int. Conf. Learning Representations (ICLR)*, 2017.',
    "ho2020": 'J. Ho, A. Jain, and P. Abbeel, "Denoising diffusion probabilistic models," in *Advances in Neural Information Processing Systems 33 (NeurIPS 2020)*, 2020, pp. 6840–6851.',
    "zhang2024tabsyn": 'H. Zhang, J. Zhang, Z. Shen, B. Srinivasan, X. Qin, C. Faloutsos, H. Rangwala, and G. Karypis, "Mixed-type tabular data synthesis with score-based diffusion in latent space," in *Proc. 12th Int. Conf. Learning Representations (ICLR)*, 2024.',
    "borisov2023": 'V. Borisov, K. Seßler, T. Leemann, M. Pawelczyk, and G. Kasneci, "Language models are realistic tabular data generators," in *Proc. 11th Int. Conf. Learning Representations (ICLR)*, 2023.',
})
REFS.update({
    "sajjadi2018": 'M. S. M. Sajjadi, O. Bachem, M. Lucic, O. Bousquet, and S. Gelly, "Assessing generative models via precision and recall," in *Advances in Neural Information Processing Systems 31 (NeurIPS 2018)*, 2018.',
    "kynkaanniemi2019": 'T. Kynkäänniemi, T. Karras, S. Laine, J. Lehtinen, and T. Aila, "Improved precision and recall metric for assessing generative models," in *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*, 2019.',
    "naeem2020": 'M. F. Naeem, S. J. Oh, Y. Uh, Y. Choi, and J. Yoo, "Reliable fidelity and diversity metrics for generative models," in *Proc. 37th Int. Conf. Machine Learning (ICML)*, PMLR vol. 119, pp. 7176–7185, 2020.',
    "alaa2022": 'A. Alaa, B. Van Breugel, E. S. Saveliev, and M. van der Schaar, "How faithful is your synthetic data? Sample-level metrics for evaluating and auditing generative models," in *Proc. 39th Int. Conf. Machine Learning (ICML)*, PMLR vol. 162, pp. 290–306, 2022.',
    "sakoe1978": 'H. Sakoe and S. Chiba, "Dynamic programming algorithm optimization for spoken word recognition," *IEEE Trans. Acoustics, Speech, and Signal Processing*, vol. 26, no. 1, pp. 43–49, 1978, doi: 10.1109/TASSP.1978.1163055.',
})
REFS.update({
    "schlegl2017": 'T. Schlegl, P. Seeböck, S. M. Waldstein, U. Schmidt-Erfurth, and G. Langs, "Unsupervised anomaly detection with generative adversarial networks to guide marker discovery," in *Information Processing in Medical Imaging (IPMI 2017)*, LNCS. Cham: Springer, 2017, pp. 146–157, doi: 10.1007/978-3-319-59050-9\\_12.',
    "schlegl2019": 'T. Schlegl, P. Seeböck, S. M. Waldstein, G. Langs, and U. Schmidt-Erfurth, "f-AnoGAN: Fast unsupervised anomaly detection with generative adversarial networks," *Medical Image Analysis*, vol. 54, pp. 30–44, 2019, doi: 10.1016/j.media.2019.01.010.',
    "akcay2018": 'S. Akcay, A. Atapour-Abarghouei, and T. P. Breckon, "GANomaly: Semi-supervised anomaly detection via adversarial training," in *Computer Vision – ACCV 2018*, LNCS 11363. Cham: Springer, 2019, pp. 622–637, doi: 10.1007/978-3-030-20893-6\\_39.',
    "zenati2018": 'H. Zenati, C. S. Foo, B. Lecouat, G. Manek, and V. R. Chandrasekhar, "Efficient GAN-based anomaly detection," arXiv preprint arXiv:1802.06222, 2018. [Preprint]',
    "donahue2017": 'J. Donahue, P. Krähenbühl, and T. Darrell, "Adversarial feature learning," in *Proc. 5th Int. Conf. Learning Representations (ICLR)*, 2017.',
})
REFS.update({
    "lin2022idsgan": 'Z. Lin, Y. Shi, and Z. Xue, "IDSGAN: Generative adversarial networks for attack generation against intrusion detection," in *Advances in Knowledge Discovery and Data Mining (PAKDD 2022)*, LNAI 13282. Cham: Springer, 2022, pp. 79–91, doi: 10.1007/978-3-031-05981-0\\_7.',
    "goodfellow2015": 'I. J. Goodfellow, J. Shlens, and C. Szegedy, "Explaining and harnessing adversarial examples," in *Proc. 3rd Int. Conf. Learning Representations (ICLR)*, 2015.',
    "madry2018": 'A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, "Towards deep learning models resistant to adversarial attacks," in *Proc. 6th Int. Conf. Learning Representations (ICLR)*, 2018.',
    "carlini2017": 'N. Carlini and D. Wagner, "Towards evaluating the robustness of neural networks," in *Proc. IEEE Symp. Security and Privacy (S&P)*, 2017, pp. 39–57, doi: 10.1109/SP.2017.49.',
    "pierazzi2020": 'F. Pierazzi, F. Pendlebury, J. Cortellazzi, and L. Cavallaro, "Intriguing properties of adversarial ML attacks in the problem space," in *Proc. IEEE Symp. Security and Privacy (S&P)*, 2020, pp. 1332–1349, doi: 10.1109/SP40000.2020.00073.',
})
REFS.update({
    "mao2019msgan": 'Q. Mao, H.-Y. Lee, H.-Y. Tseng, S. Ma, and M.-H. Yang, "Mode seeking generative adversarial networks for diverse image synthesis," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2019, pp. 1429-1437, doi: 10.1109/CVPR.2019.00152.',
    "wilkens2021": 'F. Wilkens, F. Ortmann, S. Haas, M. Vallentin, and M. Fischer, "Multi-stage attack detection via kill chain state machines," in *Proc. 3rd Workshop on Cyber-Security Arms Race (CYSARM)*, 2021, pp. 13-24, doi: 10.1145/3474374.3486918.',
})
REFS.update({
    "moustafa2015": 'N. Moustafa and J. Slay, "UNSW-NB15: A comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)," in *Proc. Military Communications and Information Systems Conf. (MilCIS)*, 2015, pp. 1-6, doi: 10.1109/MilCIS.2015.7348942.',
    "koroniotis2019": 'N. Koroniotis, N. Moustafa, E. Sitnikova, and B. Turnbull, "Towards the development of realistic botnet dataset in the Internet of Things for network forensic analytics: Bot-IoT dataset," *Future Generation Computer Systems*, vol. 100, pp. 779-796, 2019, doi: 10.1016/j.future.2019.05.041.',
    "alsaedi2020": 'A. Alsaedi, N. Moustafa, Z. Tari, A. Mahmood, and A. Anwar, "TON_IoT telemetry dataset: A new generation dataset of IoT and IIoT for data-driven intrusion detection systems," *IEEE Access*, vol. 8, pp. 165130-165150, 2020, doi: 10.1109/ACCESS.2020.3022862.',
    "neto2023": 'E. C. P. Neto, S. Dadkhah, R. Ferreira, A. Zohourian, R. Lu, and A. A. Ghorbani, "CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment," *Sensors*, vol. 23, no. 13, art. 5941, 2023, doi: 10.3390/s23135941.',
})
REFS.update({
    "demsar2006": 'J. Dem\u0161ar, "Statistical comparisons of classifiers over multiple data sets," *J. Machine Learning Research*, vol. 7, pp. 1-30, 2006.',
    "holm1979": 'S. Holm, "A simple sequentially rejective multiple test procedure," *Scandinavian J. Statistics*, vol. 6, no. 2, pp. 65-70, 1979.',
    "dietterich1998": 'T. G. Dietterich, "Approximate statistical tests for comparing supervised classification learning algorithms," *Neural Computation*, vol. 10, no. 7, pp. 1895-1923, 1998, doi: 10.1162/089976698300017197.',
})
REFS.update({
    "hayes2019logan": 'J. Hayes, L. Melis, G. Danezis, and E. De Cristofaro, "LOGAN: Membership inference attacks against generative models," *Proceedings on Privacy Enhancing Technologies*, vol. 2019, no. 1, pp. 133-152, 2019, doi: 10.2478/popets-2019-0008.',
    "hu2021tablegan_mca": 'A. Hu, R. Xie, Z. Lu, A. Hu, and M. Xue, "TableGAN-MCA: Evaluating membership collisions of GAN-synthesized tabular data releasing," in *Proc. 2021 ACM SIGSAC Conf. Computer and Communications Security (CCS)*, 2021, pp. 2096-2112, doi: 10.1145/3460120.3485251.',
    "abadi2016": 'M. Abadi, A. Chu, I. Goodfellow, H. B. McMahan, I. Mironov, K. Talwar, and L. Zhang, "Deep learning with differential privacy," in *Proc. 2016 ACM SIGSAC Conf. Computer and Communications Security (CCS)*, 2016, pp. 308-318, doi: 10.1145/2976749.2978318.',
    "jordon2019pategan": 'J. Jordon, J. Yoon, and M. van der Schaar, "PATE-GAN: Generating synthetic data with differential privacy guarantees," in *Proc. 7th Int. Conf. Learning Representations (ICLR)*, 2019.',
    "mcmahan2017": 'B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, "Communication-efficient learning of deep networks from decentralized data," in *Proc. 20th Int. Conf. Artificial Intelligence and Statistics (AISTATS)*, PMLR vol. 54, pp. 1273-1282, 2017.',
})
REFS.update({
    "arora2018": 'S. Arora, A. Risteski, and Y. Zhang, "Do GANs learn the distribution? Some theory and empirics," in *Proc. 6th Int. Conf. Learning Representations (ICLR)*, 2018.',
})
REFS.update({
    "lundberg2017": 'S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)*, 2017, pp. 4765-4774.',
    "ribeiro2016": 'M. T. Ribeiro, S. Singh, and C. Guestrin, "\\"Why should I trust you?\\": Explaining the predictions of any classifier," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining (KDD)*, 2016, pp. 1135-1144, doi: 10.1145/2939672.2939778.',
})
REFS.update({
    "gdpr2016": 'European Parliament and Council, "Regulation (EU) 2016/679 (General Data Protection Regulation)," *Official Journal of the European Union*, L 119, pp. 1-88, 2016.',
    "euaiact2024": 'European Parliament and Council, "Regulation (EU) 2024/1689 laying down harmonised rules on artificial intelligence (Artificial Intelligence Act)," *Official Journal of the European Union*, L series, 2024.',
    "stadler2022": 'T. Stadler, B. Oprisanu, and C. Troncoso, "Synthetic data -- anonymisation groundhog day," in *Proc. 31st USENIX Security Symposium*, 2022, pp. 1451-1468.',
    "dallaire2018": 'P.-L. Dallaire-Demers and N. Killoran, "Quantum generative adversarial networks," *Physical Review A*, vol. 98, no. 1, art. 012324, 2018, doi: 10.1103/PhysRevA.98.012324.',
    "lloyd2018": 'S. Lloyd and C. Weedbrook, "Quantum generative adversarial learning," *Physical Review Letters*, vol. 121, no. 4, art. 040502, 2018, doi: 10.1103/PhysRevLett.121.040502.',
})
