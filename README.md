# adversarial_data_augmentation_with_gan
This is official implementation about NeurIPS 2021 undereview paper <Domain Generalization via Adversarially Learned Novel Domains>


If you want to do experiment with digits data, please download MNIST,SVHN,MNIST-M,SYN-digits datasets to file with the same name at first.
  
If you want to train a dlow model on digits dataset with source domain mnist, targets domain mnist, svhn, mnist-m. You can go to dlow_for_digits file, then use "python3 train_dlow_digits.py --source_idx=1 --t1_idx=2 --t3_idx=3 --chechpoint_dir='name_you_want'"
   
  
  
  
  


