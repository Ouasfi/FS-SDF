# Few 'Zero Level Set'-Shot Learning of Shape Signed Distance Functions in Feature Space
  Amine Ouasfi, Adnane Boukhayma. [ECCV 2022](https://arxiv.org/abs/2207.04161) 
![](./pipe2.png)


# Install 

Please follow the instructions detailed in [IF-Net](https://github.com/Ouasfi/if-net/tree/stable). Then,  install the required  packages for this project by running:
```
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```
# Demo (New)

To run our pretrained model on your own shapes, place them in a directory and create a `.lst` file containing your file names. Then run , 
```
python demo.py -data_path [path to your data directory]  -exp <exp_name> -checkpoint <checkpoint> -inner_steps 5  -pc_samples 3000 -res 128 -batch_size 8 -save_path gen_dir/
```
where `res` is the resolution of the encoder's grid and `exp_name` is the path to the folder containing the trained model checkpoints. 
# Data Preparation

The data preparation process is the same as in [IF-Net](https://github.com/Ouasfi/if-net/tree/stable). 

To generate ground truth Signed distance values instead of occupancies run:
```
python data_processing/boundary_sampling_sdf.py -sigma 0.1
python data_processing/boundary_sampling_sdf.py -sigma 0.01
```
## Training
 To start a training please run  the following command:
 ````
 python train.py -res 128 -pc_samples 3000 -epochs 100 -inner_steps 5 -batch_size 8

  ````
  You can add the following  options `-p_enc` and `_p_dec` to initialize the encoder and/or the decoder with a pretrained model. Also, you can freeze the encoder during the training by adding the option  `-freeze` to your command. 

## Generation

The command:

````
python generate.py -res 128 -pc_samples 3000 -batch_size 8 -inner_steps 5 -exp <exp_name> -checkpoint <checkpoint>  
````
Where `exp_name` is the path to the folder containing the trained model checkpoints.

## Evaluation 
The evaluation process is the same as in [IF-Net](https://github.com/Ouasfi/if-net/tree/stable). 

## Pretrained Models

Pretrained models can be found [here](https://drive.google.com/drive/folders/1ACokEAU91cuSaS4C8qdfjHo1KZV3hbuz?usp=sharing).

## Contact

For questions and comments regarding the code please contact [Amine Ouasfi]() via mail. (See Paper)

