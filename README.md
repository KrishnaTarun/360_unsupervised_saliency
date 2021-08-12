# Rethinking 360 Image Visual Attention Modelling with Unsupervised Learning.


### Contrastive Learning

In order to train the Encoder suitable for the downstream task, run:

    cd contrastive/
    python main.py --batch_size 20 --nce_k 16000 --print_freq 100 --data_folder data --dataset Train
    
#### Dataset

The unlabelled data used for training encoder and as reported in the paper can be downloaded from here [click](https://drive.google.com/file/d/1QtxRurC0ac5Aksywg1XDRfUMg64fEIqr/view?usp=sharing). Make sure to follow this hierarchy for data: data/Train/images/ or otherwise change the code accordingly. Under *images* you have all your jpg images.

#### Pre-Trained models
 - [VGG](https://drive.google.com/file/d/1f-nk2O66sZc-LVpi2o8tJkV1yQhztEtg/view?usp=sharing)
 - [ResNet](https://drive.google.com/file/d/1VQY1KKPL5gBq6gycZmDC6j0WgDB6Xqot/view?usp=sharing)
 
 
     
    
