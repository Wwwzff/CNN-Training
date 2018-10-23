# MiniProject2
This project mainly focus on the beginning methods of the machine learning. Given a picture and predict whether it's a cat or dog. Contact with chesterw@bu.edu or wechatID Wwwwzf- if having trouble with code reviewing stuffs. 
## Dataset
Dataset can be downloaded from kaggle in the following link:
```
https://www.kaggle.com/c/dogs-vs-cats/data
```
## Running Environment
- ubuntu 18.04
- tensorflow-gpu with GTX1070
- jupyter notebook

## Cnn
### Reference
Methods found on CSDN in the following link and modified by chester@bu.edu
```
https://blog.csdn.net/Mbx8X9u/article/details/79124840?utm_source=blogxgwz0
```
### Hyper Parameters
- batch size 32
- learning rate 0.01
- epoch (at most) 50
- resize pics to 150*150
### Early Stopping
Training will be stopped if validation loss doesn't change for 2 consecutive epochs 



## 3model.py
There're three models inplemented in the python script, and you can use the file structure set by the cnn.py and train directly on your trainingset

Resize the pic to 197*197, which should be concerned when using test.py to test your own pics (resize your own pics to fit with the model)
## Result
* Training result can be found in the .html file 
* Trained models are too big to upload, but can be downloaded using link below (need bu email):

```
https://drive.google.com/file/d/1R-Ab3tGMXtZ_pk7sNLJQpEmFDUH9BNfn/view?usp=sharing
```

* Result from the tester are showed below:
<img src="https://github.com/Wwwzff/MiniProject2/blob/master/results/test_result.png" />
