### Basic Information:
This code is released for the paper: 

Lewu, Peijie Sun, Yanjie Fu, Richang Hong, Xiting Wang and Meng Wang. A Neural Influence Diffusion Model for Social Recommendation. Accepted by SIGIR2019. 

You can also preview our paper from the [arXiv](https://arxiv.org/submit/2661010/view).

### Usage:
1. Environment: I have tested this code with python2.7, tensorflow-gpu-1.12.0 
2. Download the yelp data from this [link](https://drive.google.com/drive/folders/1hIkRDIVI87CUM4xFGjHMeipOlPz97ThX?usp=sharing), and unzip the directories in yelp data to the root directory of your local clone repository.
3. cd the diffnet directory and execute the command `python entry.py --data_name=<data_name> --model_name=<model_name> --gpu=<gpu id>` then you can see it works, if you have any available gpu device, you can specify the gpu id, or you can just ignore the gpu id. 

Following is an example:
`python entry.py --data_name=yelp --model_name=diffnet`

### Author contact:
Email: sun.hfut@gmail.com
