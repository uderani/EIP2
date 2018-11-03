#### Results:

![](https://media.giphy.com/media/AuwBPJztsEWkw/giphy.gif)



| Accuracy |                            Params                            |      |
| :------: | :----------------------------------------------------------: | :--: |
|  92.74   | 913k(3 dense blocks, depth 48, growth in each dense block: 16, 32, 24) |      |
|   91.3   |               830k(4 dense blocks w/o wingrad)               |      |
|   91.2   |              990k(4 dense blocks with winograd)              |      |
|   ~88    |             0.8M (Many unsuccessfully attempts)              |      |



#### The approaches taken/understood while trying to get the required valition accuracy:

![](https://media.giphy.com/media/g1a84q6RBSMrS/giphy.gif)



- Reducing the number of **dense blocks** (~~four~~) to **three** since the feature map gets very small till it reaches the fourth block
- Adopt **more filters** and <u>less depth(40-48)</u> initially since increasing number of layers and dense block increase <u>compute requirements</u>(<u>have to decrease batch size a lot</u>) 
- Keeping <u>learning rate</u> high more for most number of epochs and then slow close in
- Use <u>smaller images 24x24 to train faster</u> and hopefully more important features and then train on 32x32
- Use <u>Global Averaging Pool</u> to train on multiple size images instead of Flatten
- Used combinations of multiple <u>Augmentation techniques</u>(rotation:16, width shift: 0.125, horizontal flip)
- <u>Winograd's minimal filtering algorithm</u> to have more filters, reduce params and execution time
- Use <u>Global Contrast Normalization</u>. What I saw specially for 24x24 is that the borders were more clearly visible and not merge(mixed) into each other thus giving sharper features
- Used dropout in dense block as well



#### **Things learnt from class used here**

![](https://media.giphy.com/media/AYhNomPYtxFOU/giphy.gif)

-  **Winograd filtering algortithm**[ (3x1) followed by (1x3)]
- **Data is very important**: Importance of varied data from existing to train better(dat augmentation)
- How **SGD converges visually** and thus reducing lr to make the rolling ball move slowly at the end
- **Normalize data** (Assignment 3)
- Regularization using **<u>dropout<**/u> and image aigmentation
- **1x1** Conv to **reduce inputs** also mentioned as Bottleneck layer in paper

#### Things learnt from paper:

![](https://media.giphy.com/media/8dYmJ6Buo3lYY/giphy.gif)



- Adopt **Bottleneck layers</u>** which allowed to have more params with same depth and filters
- **Random cropping**(width shift) 4 pixel padding 4/32= 0.125 and also **Horizontal Flip**
- <u>Global Average Pooling</u> 
- 0.2 Dropout and 0.5 Compression
- <u>DenseNet-BC</u> has better peformance with **less params** <u>than DenseNet</u>

#### Architecture

------

​										Input (None, None, 3)			

------

Convolution  							(5*1,64)

​										(1*5,64)

------

Dense Block								[1*1] x 16

​	(1)									[3*1] x 16

​										[1*3] x 16

------

Transition Layer							[1*1] x 16

​	(1)									Avergaing Pool

------

Dense Block								[1*1] x 32

​	(2)									[3*1] x 32

​										[1*3] x 32

------

Transition Layer							[1*1] x 16

​	(2)									Avergaing Pool

------

Dense Block								[1*1] x 24

​	(3)									[3*1] x 24

​										[1*3] x 24

------

Classifiction	 						GlobalAveragePooling2D

​     Layer								10(number of classes)D Fully Connected

------



#### And finally the code:

![](https://media.giphy.com/media/ZVik7pBtu9dNS/giphy.gif)



