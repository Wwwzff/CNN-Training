# MiniProject2
This project mainly focus on the beginning methods of the machine learning. Given a picture and predict whether it's a cat or dog.
## Dataset
Dataset can be download from kaggle in the following link:
```
https://www.kaggle.com/c/dogs-vs-cats/data
```
## Running Environment
- ubuntu 18.04
- tensorflow-gpu with GTX1070
- jupyter notebook

## Cnn
### Reference&Dataset
Methods found on CSDN in the following link and modified by chester@bu.edu
```
https://blog.csdn.net/Mbx8X9u/article/details/79124840?utm_source=blogxgwz0
```
### Hyper Parameters
- batch size 32
- learning rate 0.01
- epoch (at most) 50

### Early Stopping
Training will be stopped if validation loss doesn't change for 2 consecutive epochs 


### result
You can find result in the file called CNN-Result.html
