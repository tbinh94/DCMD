# [Dual Conditioned Motion Diffusion for Pose-Based Video Anomaly Detection](https://ojs.aaai.org/index.php/AAAI/article/view/32829), [Arxiv](https://arxiv.org/abs/2412.17210)

## Abstract
Video Anomaly Detection (VAD) is essential for computer vision research. Existing VAD methods utilize either reconstruction-based or prediction-based frameworks. The former excels at detecting irregular patterns or structures, whereas the latter is capable of spotting abnormal deviations or trends. We address pose-based video anomaly detection and introduce a novel framework called Dual Conditioned Motion Diffusion (DCMD), which enjoys the advantages of both approaches. The DCMD integrates conditioned motion and conditioned embedding to comprehensively utilize the pose characteristics and latent semantics of observed movements, respectively. In the reverse diffusion process, a motion transformer is proposed to capture potential correlations from multi-layered characteristics within the spectrum space of human motion. To enhance the discriminability between normal and abnormal instances, we design a novel United Association Discrepancy (UAD) regularization that primarily relies on a Gaussian kernel-based time association and a self-attention-based global association. Finally, a mask completion strategy is introduced during the inference stage of the reverse diffusion process to enhance the utilization of conditioned motion for the prediction branch of anomaly detection. Extensive experiments conducted on four datasets demonstrate that our method dramatically outperforms state-of-the-art methods and exhibits superior generalization performance.

![image](https://github.com/guijiejie/DCMD-main/blob/main/architecture.png)
## Content
```
.
├── config
│   ├── Avenue
│   │   ├── dcmd_test.yaml
│   │   └── dcmd_train.yaml
│   ├── STC
│   │   ├── dcmd_test.yaml
│   │   └── dcmd_train.yaml
│   └── UBnormal
│       ├── dcmd_test.yaml
│       └── dcmd_train.yaml
├── environment.yaml
├── eval_DCMD.py
├── models
│   ├── common
│   │   └── components.py
│   ├── gcae
│   │   └── stsgcn.py
│   ├── dcmd.py
│   ├── transformer.py
│   ├── stsae
│   │    ├── stsae.py
│   │    └── stsae_unet.py
│   └── transformer.py
├── README.md
├── train_DCMD.py
└── utils
    ├── argparser.py
    ├── data.py
    ├── dataset.py
    ├── dataset_utils.py
    ├── diffusion_utils.py
    ├── ema.py
    ├── eval_utils.py
    ├── get_robust_data.py
    ├── __init__.py
    ├── model_utils.py
    ├── preprocessing.py
    └── tools.py
    
```

  </div>
</section>
