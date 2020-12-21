### Basic Information:
This code is released for the paper: 

Le Wu, Junwei Li, Peijie Sun, Richang Hong, Yong Ge, and Meng Wang. DiffNet++: A Neural Influence and Interest Diffusion Network for Social Recommendation. Accepted by IEEE Transactions on Knowledge and Data Engineering in Dec 2020. 

DiffNet++ is improved on [DiffNet](https://github.com/PeiJieSun/diffnet), for more details, please refer to the paper from the [arXiv](https://arxiv.org/abs/2002.00844), the final version will be updated soon.

### Usage:
1. Environment: python2.7, tensorflow-gpu-1.12.0
2. Download datasets from this [link](https://drive.google.com/drive/folders/1YAJvgsCJLKDFPVFMX3OG7v3m1LAYZD5R?usp=sharing), and just put the downloaded folder 'data' in the root directory of 'DiffNet++'. 
3. Start with the command `python entry.py --data_name=<data_name> --model_name=<model_name> --gpu=<gpu id>`, any available gpu device can be used by specifying the gpu id, or you can just ignore the gpu id. 


Example:
`python entry.py --data_name=yelp --model_name=diffnetplus`


### Citation:
```

 This code corresponds to DiffNet++, The texts about the citation details (e.g., volume and/or issue number, publication year, and page numbers) will be changed and updated when they are available.
 @article{wu2020diffnet++,
  title={DiffNet++: A Neural Influence and Interest Diffusion Network for Social Recommendation},
  author={Wu, Le and Li, Junwei and Sun, Peijie and Ge, Yong and Wang, Meng},
  journal={arXiv preprint arXiv:2002.00844},
  year={2020}
 }


 The dataset flickr we used in DiffNet++ is provided by:
 @article{leHASC,
  title={A Hierarchical Attention Model for Social Contextual Image Recommendation},
  author={Wu,Le and Chen, Lei and Hong, Richang and Fu, Yanjie, and Xie, Xing and Wang, Meng},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  pages={1854--1867},
  year={2019}
 }


 DiffNet++ is improved on DiffNet:
 @inproceedings{DiffNET,
 title={A Neural Influence Diffusion Model for Social Recommendation},
 author={Wu, Le and Sun, Peijie, and Fu, Yanjie and Hong, Richang, and Wang, Xiting and Wang, Meng},
 booktitle={42nd International ACM SIGIR Conference on Research and Development in Information Retrieval},
 pages={235–-244},
 year={2019}
 }
 
 ```

### Author contact:
Free feel to contact us if you have any questions.

Email: lijunwei.edu@gmail.com, sun.hfut@gmail.com
