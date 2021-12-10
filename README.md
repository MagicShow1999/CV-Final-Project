# CV-Final-Project
The baseline model is pretrained VGG16 or VGG19. Then we use our customized model to return some layers that represent content or style.
This is a simple working version of style transfer using VGG19.
https://colab.research.google.com/drive/1mk-e8o80cBHKzURTXZqYCn3KCqMxOZ9g#scrollTo=UlQXHzg6n3zq
The code is from https://github.com/AyushExel/Neural-Style-Transfer/blob/master/styleTransfer.py
And there's a tutorial about it https://www.youtube.com/watch?v=K_xBhp1YsrE&list=PLbMqOoYQ3MxwV-xLpzWNQ70IvctU7H-yl&index=2

Another version of style transfer using VGG19 and it has a very detailed blog explaining concepts (implemented with Keras):
https://medium.com/tensorflow/neural-style-transfer-creating-art-with-deep-learning-using-tf-keras-and-eager-execution-7d541ac31398
(can rephrase a lot of the content into our project lol)

## Trying different models
1. VGG19 - most popular because it is good at extracting content and style
2. VGG16 
3. GAN?? (not sure..)

## Advanced topics: Realtime style transfer (video)
Using MSGNet to achieve a realtime style transfer: https://github.com/StacyYang/MSG-Net


## Our goal
1. Make a working version of style transfer with different trials of pretrained models
2. Adjust hyperparameters to compare results
3. Try to implement a realtime style transfer model 

