<h1>MFE - Surface-based Multimodal Protein-Ligand Binding
Aﬀinity Prediction</h1> 



<h2>Install</h2>

```
git clone git@github.com:Sultans0fSwing/MFE.git
cd MFE
conda env create -f environment.yaml
```

<h2>Dataset</h2>

We use the PDBbind dataset (version 2016) as the dataset for the protein-ligand binding affinity prediction task. The official original dataset can be downloaded from http://pdbbind.org.cn/. Then, place the downloaded original dataset in the data directory.

<h2>Process Data</h2>

Before starting the training, please use `process.py` to transform the raw data into the format required by the model. Remember, you need to specify the directories for the training set, validation set, and test set yourself.

<h2>Training</h2>

After processing the dataset, please use `train_lba.py` for training. Set the number of training epochs to `50`, the learning rate to `0.0001`, and fix the regularization coefficient at `5e-4`. The output of the model can be found in the output directory.

