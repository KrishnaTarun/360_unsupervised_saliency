# Rethinking 360 Image Visual Attention Modelling with Unsupervised Learning (ICCV 2021).


### Contrastive Learning

In order to train the Encoder suitable for the downstream task, run:

    cd contrastive/
    python main.py --batch_size 20 --nce_k 16000 --print_freq 100 --data_folder data --dataset Train
    
#### Dataset

The unlabelled data used for training encoder and as reported in the paper can be downloaded from here [click](https://drive.google.com/file/d/1QtxRurC0ac5Aksywg1XDRfUMg64fEIqr/view?usp=sharing). Make sure to follow this hierarchy for data: data/Train/images/ or otherwise change the code accordingly. Under *images* you have all your .jpg images.

#### Pre-Trained models
 - [VGG](https://drive.google.com/file/d/1f-nk2O66sZc-LVpi2o8tJkV1yQhztEtg/view?usp=sharing)
 - [ResNet](https://drive.google.com/file/d/1VQY1KKPL5gBq6gycZmDC6j0WgDB6Xqot/view?usp=sharing)
 
 #### Pre-Trained saliency models
 - [SalGAN based](https://drive.google.com/drive/folders/1EwJHC0xgGVCXdOTaBDKyQJ6vHbe98tBp?usp=sharing)
 - [Random initialization](https://drive.google.com/drive/folders/1pwpr4NIxG24i-OukJ16kJWD4GuWpBpep?usp=sharing)
 
 #### Please consider citing the following paper if you find this work useful for your research.
 
    @InProceedings{Djilali_2021_ICCV,
    author    = {Djilali, Yasser Abdelaziz Dahou and Krishna, Tarun and McGuinness, Kevin and O'Connor, Noel E.},
    title     = {Rethinking 360deg Image Visual Attention Modelling With Unsupervised Learning.},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {15414-15424}
    }
 
     
    
