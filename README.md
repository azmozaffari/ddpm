# DDPM

This is the implementation of [Denoising Diffusion Probabilistic Model](https://arxiv.org/abs/2006.11239) in PyTorch with training and inference parts from scratch.


![plot](./images/denoising-diffusion.png)


# Quick Start
* Create a new conda environment with python 3.8 then run below commands

* git clone https://github.com/azmozaffari/ddpm.git

* cd ddpm

* pip install -r requirements.txt

* For training/inference use the below commands passing the desired configuration file as the config argument in case you want to play with it.

* python3 main.py --config "path/to/config/file" --mode training       for training ddpm

* python3 main.py --config "path/to/config/file" --mode inference      for generating images

# Results
The trained checkpoints will be saved in checkpoints folder during the training.

The generated images are saved in outcomes folder after running the inference mode.

Here are some results generated respecting to MNIST dataset:

<table>
  <tr>
    <td> <img src="./images/1.jpg"  alt="1" width = 36px height = 36px ></td>
    <td> <img src="./images/111.jpg"  alt="2" width = 36px height = 36px ></td>    
    <td> <img src="./images/221.jpg" alt="3" width = 36px height = 36px ></td>    
    <td> <img src="./images/331.jpg" alt="4" width = 36px height = 36px ></td>    
    <td> <img src="./images/441.jpg" alt="5" width = 36px height = 36px ></td>    
    <td> <img src="./images/551.jpg" alt="6" width = 36px height = 36px ></td>    
    <td> <img src="./images/661.jpg" alt="7" width = 36px height = 36px ></td>    
    <td> <img src="./images/771.jpg" alt="8" width = 36px height = 36px ></td>    
    <td> <img src="./images/881.jpg" alt="9" width = 36px height = 36px ></td>    
    <td> <img src="./images/991.jpg" alt="10" width = 36px height = 36px ></td>
  </tr> 
  <tr>
    <td> <img src="./images/10.jpg"  alt="1" width = 36px height = 36px ></td>
    <td> <img src="./images/1110.jpg"  alt="2" width = 36px height = 36px ></td>    
    <td> <img src="./images/2210.jpg" alt="3" width = 36px height = 36px ></td>    
    <td> <img src="./images/3310.jpg" alt="4" width = 36px height = 36px ></td>    
    <td> <img src="./images/4410.jpg" alt="5" width = 36px height = 36px ></td>    
    <td> <img src="./images/5510.jpg" alt="6" width = 36px height = 36px ></td>    
    <td> <img src="./images/6610.jpg" alt="7" width = 36px height = 36px ></td>    
    <td> <img src="./images/7710.jpg" alt="8" width = 36px height = 36px ></td>    
    <td> <img src="./images/8810.jpg" alt="9" width = 36px height = 36px ></td>    
    <td> <img src="./images/9910.jpg" alt="10" width = 36px height = 36px ></td>
  </tr> 
</table>


# Configuration
* config.yml - Allows you to play with different components of ddpm
