# HR-Morphed-face-images-detection

## This is a repository to conduct the experiments for morphed images detection based on the given eye photo. 

### 1. Environment installation
```
conda create --name morph python=3.10
conda activate morph
pip install -r requirements.txt
```

### 2. Memory map usage
In this repository, it is possible to use the memory map file (because of possible system memory restrictions). To create a np.memmap file from the directory with images, use ```code/create_memmap.py``` script. 

### 3. Training configs description

There are two configs available for the training ```code/configs/config.json``` for a training with dataset stored as collection of images in a directory (original version), or ```code/configs/config_memmap.json``` for a training using data in a format of np.memmap object. Refer to existing versions for the examples of config fields values. 

For ```model_name``` field following values are available: 
1. debugnn
2. s2dcnn
3. efficientnet_b0
4. efficientnet_b1
5. resnet18
6. resnet34
7. mobilenet_v3_small

Refer to the report provided in the repository for the description of the DebugNN and S2DCNN architechtures. 

For ```scheduler``` value use one of the following: 
1. CosineAnnealingLR
2. ReduceLROnPlateau
3. StepLR

All the schedulers are the default implementations of torch.optim framework. 

For ```optimizer``` value use one of the following: 
1. adam
2. sgd


### 4. Launch training
Use the following command to launch the training using a dataset stored as a collection of images (default version): 
```
python run.py train --config configs/config.json
```

Use the following command to launch the training using a dataset stored as a collection of images (default version): 
```
python run.py train --config configs/config_memmap.json --memmap
```

### 5. Model Evaluation

After training, the performance of the model can be evaluated using two different testing methods:

1. **Per-class evaluation**  
   Conducted using `code/test_section.py`, this method computes test metrics separately for each morph class. The results are stored in `results/evaluation_section.json`.

```
python test_section.py --config configs/config.json --datadir path/to/dataset --checkpoint path/to/checkpoint.pth --model_name <architecture_name> --batch_size <batch_size> --name <name_in_json>
```

2. **Overall evaluation**  
   Conducted using `code/test_total.py`, this method evaluates the model on the entire dataset as a whole. The results are stored in `results/evaluation_total.json`.

```
python test_total.py --config configs/config.json --datadir path/to/dataset --checkpoint path/to/checkpoint.pth --model_name <architecture_name> --batch_size <batch_size> --name <name_in_json>
```

Replace `<architecture_name>` with the architecture used for training and `<name_in_json>` is the name used in the results json file.

For a visual analysis of the evaluation results, refer to `results/plot.ipynb`, which generates plots for better interpretation of model performance.

Additionally, `code/test_old.py`, `results/evaluation_old.json` and `results/plot_old.ipynb` are available, which were used mid-research for the initial experiments and results. These files correspond to an earlier phase of the study (and may utilize different models).

```
python test_old.py --config configs/config.json --datadir path/to/dataset --checkpoint path/to/checkpoint.pth --model_name <model_name> --batch_size <batch_size>
```