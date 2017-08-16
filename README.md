# DREAM

This repository contains my implementations of DREAM for next basket prediction. Besides I extendted the DREAM Framework to reorder prediction scenario. And it helped me earn 39/2669 place in Kaggle Instacart Reorder Prediction Competition. For anyone who is interested, please check [this page](https://www.kaggle.com/c/instacart-market-basket-analysis) for details about the Instacart competition.

## Model

DREAM uses RNN to capure sequential information of users' shopping behavior. It extracts users' dynamic representations and scores user-item pair by calculating inner products between users' dynamic representations and items' embedding.  

Refer to the following paper:

> Yu, Feng, et al. "A dynamic recurrent model for next basket recommendation." Proceedings of the 39th International ACM SIGIR conference on Research and Development in Information Retrieval. ACM, 2016.

for details about DREAM.


## Dataset 

It runs on [the Instacart dataset](https://tech.instacart.com/3-million-instacart-orders-open-sourced-d40d29ead6f2) and can be used in other e-commerce datasets by modifying the input easily. 

## Files

> `dream.py`
> * definition of DREAM
> `train.py`
> * implementation of bpr loss function
> * implemeantation of reorder bpr loss function
> * training of DREAM
> `eval.py`
> * calculate <u,p> score using DREAM
> `data.py`
> * input wrapper for DREAM
> * based on the Instacart Dataset
> `utils.py`
> * some useful functions
> `config.py`
> * DREAM configurations
> `constants.py`	
> * some constants such as file path
> `Make Recommendation Using DREAM`
> * using trained DREAM model to generate predictors for <u,p> 

## Requirements

- pytorch == 0.1.12.post2
- pandas ==  0.19.2
- scikit-learn == 0.18.1

You need GPU to accelerate training.

## License

Copyright (c) 2017 Yihong Chen