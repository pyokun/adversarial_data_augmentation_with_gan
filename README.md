# Domain Generalization via Adversarially Learned Novel Domains
This is official implementation about NeurIPS 2021 undereview paper "Domain Generalization via Adversarially Learned Novel Domains"


If you want to do experiment with digits data, please download MNIST,SVHN,MNIST-M,SYN-digits datasets to file with the same name at first.
  
To implement our proposal, first, you need to train a dlow model on digits dataset with source domain mnist, targets domain mnist, svhn, mnist-m. You can go to dlow_for_digits file, then use "python3 train_dlow_digits.py --source_idx=1 --t1_idx=2 --t3_idx=3 --chechpoint_dir='name_of_saved_dlow_model'"
   
Then you can go to digits_classification/scripts dir, and use "python3 train_digits_classifier.py --source_idx=1 --additional_idx1=2  --additional_idx2=3 --test_idx=4 --dlow_name="generator/name_of_saved_dlow_model" --output_dir=result" to train a classifier on given training domains, and evaluate on a test domain. This part is based on domainbed https://github.com/facebookresearch/DomainBed
 
 

If you want to do experiment with PACS dataset, please download PACS dataset by yourself. 
  
First, you need to train a dlow model, you can go dlow_for_pacs file, then use python3 train.py --s_idx='p' --t1_idx='a' --t2_idx='c' --checkpoint_dir='name_of_saved_dlow_model'.  Here, 'p','a','c' means photo domain, art domain, cartoon domain.
  
  
  


