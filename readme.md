### ⭐UPDATE 20221220:
*As the python 2.7 is deprecated, I have convert the diffnet code into a new one to make it can be used under python 3.x. If you use python 3.x, tensorflow-gpu-1.x, you can run the code in directory diffnet-tensorflow-v1-python3. I have tested the development environment python 3.7, and tensorflow-1.15.*

### Basic Information:
This code is released for the papers: 

Le Wu, Peijie Sun, Yanjie Fu, Richang Hong, Xiting Wang and Meng Wang. A Neural Influence Diffusion Model for Social Recommendation. Accepted by SIGIR2019. [pdf](http://arxiv.org/abs/1904.10322).  
Le Wu, Junwei Li, Peijie Sun, Richang Hong, Yong Ge, and Meng Wang. DiffNet++: A Neural Influence and Interest Diffusion Network for Social Recommendation. Accepted by IEEE Transactions on Knowledge and Data Engineering in Dec 2020. [pdf](https://arxiv.org/abs/2002.00844)


### Usage:
1. **Environment: If you use python2.7, tensorflow-gpu-1.12.0, you can run the code in directory diffnet-tensorflow-v1; if you use python 3.7, tensorflow-gpu-1.15, you can run the code in directory diffnet-tensorflow-v1-python3.**
3. Run DiffNet: 
   1. Download the yelp data from this [link](https://drive.google.com/drive/folders/1hIkRDIVI87CUM4xFGjHMeipOlPz97ThX?usp=sharing), and unzip the directories in yelp data to the sub-directory named diffnet of your local clone repository.
   2. cd the sub-directory diffnet and execute the command `python entry.py --data_name=<data_name> --model_name=diffnet --gpu=<gpu id>` 
4. Run DiffNet++:
   1. Download datasets from this [link](https://drive.google.com/drive/folders/1YAJvgsCJLKDFPVFMX3OG7v3m1LAYZD5R?usp=sharing), and just put the downloaded folder 'data' in the sub-directory named diffnet++ of your local clone repository.
   2. cd the sub-directory diffnet++ and execute the command `python entry.py --data_name=<data_name> --model_name=diffnetplus --gpu=<gpu id>` 
5. If you have any available gpu device, you can specify the gpu id, or you can just ignore the gpu id. 

Following are the command examples:  
`python entry.py --data_name=yelp --model_name=diffnet`  
`python entry.py --data_name=yelp --model_name=diffnetplus`

### Citation:
```
The dataset flickr we use from this paper:
 @article{HASC2019,
  title={A Hierarchical Attention Model for Social Contextual Image Recommendation},
  author={Le, Wu and Lei, Chen and Richang, Hong and Yanjie, Fu and Xing, Xie and Meng, Wang},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2019}
 }

 The algorithm is from DiffNet and DiffNet++:
 @inproceedings{DiffNet2019.
 title={A Neural Influence Diffusion Model for Social Recommendation},
 author={Le Wu, Peijie Sun, Yanjie Fu, Richang Hong, Xiting Wang and Meng Wang},
 conference={42nd International ACM SIGIR Conference on Research and Development in Information Retrieval},
 year={2019}
 }

 @article{wu2020diffnet++,
  title={DiffNet++: A Neural Influence and Interest Diffusion Network for Social Recommendation},
  author={Wu, Le and Li, Junwei and Sun, Peijie and Ge, Yong and Wang, Meng},
  journal={arXiv preprint arXiv:2002.00844},
  year={2020}
 }
 
 We utilized the key technique in following paper to tackle the graph oversmoothing issue, and we have annotated
 the change in line 114 in diffnet/diffnet.py, if you want to konw more details, please refer to:
 @inproceedings{
 title={Revisiting Graph based Collaborative Filtering: A Linear Residual Graph Convolutional Network Approach},
 author={Lei Chen, Le Wu, Richang Hong, Kun Zhang, Meng Wang},
 conference={The 34th AAAI Conference on Artificial Intelligence (AAAI 2020)},
 year={2020}
 }
 ```

### Author contact:
Email: sun.hfut@gmail.com, lijunwei.edu@gmail.com
