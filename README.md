# GMVCN-master
PyTorch implementation of "Exploring Graph-aware Multi-View Fusion for Rumor Detection on Social Media"

## Datasets
The two datasets we used are public datasets. You can obtain the data from corresponding official links.
[Semeval 2017]
https://alt.qcri.org/semeval2017/task8/index.php?id=data-and-tools
The dataset supports stance classification (task A) and rumor detection (task B). In this paper, we focus on the task of rumor detection, and the stance labels are only used to visualize user stance in a post. 
[PHEME]
https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078

## Dependencies
python==3.8.8
torch==1.8.1
torch_geometric==2.0.1
torch_cluster==1.5.9
torch_spline_conv==1.2.1
torch_scatter==2.0.8
torch_sparse==0.6.12
matplotlib==3.5.1 
packaging==21.3
tabulate==0.8.9

