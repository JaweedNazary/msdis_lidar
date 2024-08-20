<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork/images/IDSN-logo.png" width="300" alt="cognitiveclass.ai logo">

<h1 align="center"><font size="5">AUTOENCODERS</font></h1>


Estimated time needed: **25** minutes


<div class="alert alert-block alert-info" style="margin-top: 20px">
Welcome to this notebook about autoencoders.
<font size="3"><strong>In this notebook you will learn the definition of an autoencoder, how it works, and see an implementation in TensorFlow.</strong></font>
<br>
<br>
<h2>Table of Contents</h2>
<ol>
 <li><a href="#ref1">Introduction</a></li>
 <li><a href="#ref2">Feature Extraction and Dimensionality Reduction</a></li>
 <li><a href="#ref3">Autoencoder Structure</a></li>
 <li><a href="#ref4">Performance</a></li>
 <li><a href="#ref5">Training: Loss Function</a></li>
 <li><a href="#ref6">Code</a></li>
</ol>
</div>
<br>
By the end of this notebook, you should be able to create simple autoencoders apply them to problems in the field of unsupervised learning.
<br>
<p></p>
<hr>


<a id="ref1"></a>

<h2>Introduction</h2>
An autoencoder, also known as autoassociator or Diabolo networks, is an artificial neural network employed to recreate the given input.
It takes a set of <b>unlabeled</b> inputs, encodes them and then tries to extract the most valuable information from them.
They are used for feature extraction, learning generative models of data, dimensionality reduction and can be used for compression. 

A 2006 paper named <b><a href="https://www.cs.toronto.edu/~hinton/science.pdf?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0120ENSkillsNetwork954-2023-01-01">Reducing the Dimensionality of Data with Neural Networks</a>, done by G. E. Hinton and R. R. Salakhutdinov</b>, showed better results than years of refining other types of network, and was a breakthrough in the field of Neural Networks, a field that was "stagnant" for 10 years.

Now, autoencoders, based on Restricted Boltzmann Machines, are employed in some of the largest deep learning applications. They are the building blocks of Deep Belief Networks (DBN).

<center><img src="https://ibm.box.com/shared/static/xlkv9v7xzxhjww681dq3h1pydxcm4ktp.png" style="width: 350px;"></center>


<hr>


<a id="ref2"></a>

<h2>Feature Extraction and Dimensionality Reduction</h2>

An example given by Nikhil Buduma in KdNuggets (<a href="http://www.kdnuggets.com/2015/03/deep-learning-curse-dimensionality-autoencoders.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0120ENSkillsNetwork954-2023-01-01">link</a>) gives an excellent explanation of the utility of this type of Neural Network.

Say that you want to extract the emotion that a person in a photograph is feeling. Take the following 256x256 pixel grayscale picture as an example:

<img src="https://ibm.box.com/shared/static/r5knpow4bk2farlvxia71e9jp2f2u126.png">

If we just use the raw image, we have too many dimensions to analyze.  This image is 256x256 pixels, which corresponds to an input vector of 65536 dimensions! Conventional cell phones can produce images in the  4000 x 3000 pixels range, which gives us 12 million dimensions to analyze.

This is particularly problematic, since the difficulty of a machine learning problem is vastly increased as more dimensions are involved. According to a 1982 study by C.J. Stone (<a href="http://www-personal.umich.edu/~jizhu/jizhu/wuke/Stone-AoS82.pdf?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0120ENSkillsNetwork954-2023-01-01">link</a>), the time to fit a model, is optimal if:

<br><br>

<div class="alert alert-block alert-info" style="margin-top: 20px">
<h3><strong>$$m^{-p/(2p+d)}$$</strong></h3>
<br>
Where:
<br>
m: Number of data points
<br>
d: Dimensionality of the data
<br>
p: Number of Parameters in the model
</div>

As you can see, it increases exponentially!

Returning to our example, we don't need to use all of the 65,536 dimensions to classify an emotion.
A human identifies emotions according to specific facial expressions, and some <b>key features</b>, like the shape of the mouth and eyebrows.

<center><img src="https://ibm.box.com/shared/static/m8urvuqujkt2vt1ru1fnslzh24pv7hn4.png" height="256" width="256"></center>


<hr>


<a id="ref3"></a>

<h2>Autoencoder Structure</h2>

<img src="https://ibm.box.com/shared/static/no7omt2jhqvv7uuls7ihnzikyl9ysnfp.png" style="width: 400px;">

An autoencoder can be divided in two parts, the <b>encoder</b> and the <b>decoder</b>.

The encoder needs to compress the representation of an input. In this case, we are going to reduce the dimensions of the image of the example face from 2000 dimensions to only 30 dimensions.  We will acomplish this by running the data through the layers of our encoder.

The decoder works like encoder network in reverse. It works to recreate the input as closely as possible.  The training procedure produces at the center of the network a compressed, low dimensional representation that can be decoded to obtain the higher dimensional representation with minimal loss of information between the input and the output.


<hr>


<a id="ref4"></a>

<h2>Performance</h2>

After training has been completed, you can use the encoded data as a reliable low dimensional representation of the data.  This can be applied to many problems where dimensionality reduction seems appropriate.

<img src="https://ibm.box.com/shared/static/yt3xyon4g2jyw1w9qup1mvx7cgh28l64.png">

This image was extracted from the G. E. Hinton and R. R. Salakhutdinovcomparing's <a href="https://www.cs.toronto.edu/~hinton/science.pdf?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0120ENSkillsNetwork954-2023-01-01">paper</a>, on the two-dimensional reduction for 500 digits of the MNIST, with PCA (Principal Component Analysis) on the left and autoencoder on the right. We can see that the autoencoder provided us with a better separation of data.


<hr>


<a id="ref5"></a>

<h2>Training: Loss function</h2>

An autoencoder uses the <b>Loss</b> function to properly train the network. The Loss function will calculate the differences between our output and the expected results. After that, we can minimize this error with gradient descent. There are many types of Loss functions, and it is important to consider the type of problem (classification, regression, etc.) when choosing this funtion.


<h3>Binary Values:</h3>
$$L(W) = - \sum_{k} (x_k log(\hat{x}_k) + (1 - x_k) \log (1 - \hat{x}_k) \ )$$


For binary values, we can use an equation based on the sum of Bernoulli's cross-entropy.  This loss function is best for binary classification problems.

$x_k$ is one of our inputs and $\hat{x}_k$ is the respective output.  Note that:

$$\hat{x} = f(x,W)$$

where $W$ is the full parameter set of the neural network.

We use this function so that when $x_k=1$, we want the calculated value of $\hat{x}_k$ to be very close to one, and likewise if $x_k=0$.

If the value is one, we just need to calculate the first part of the formula, that is, $-x_k log(\hat{x}_k)$. Which, turns out to just calculate $- log(\hat{x}_k)$.  We explicitly exclude the second term to avoid numerical difficulties when computing the logarithm of very small numbers.

Likewise, if the value is zero, we need to calculate just the second part, $(1 - x_k) \log (1 - \hat{x}_k))$ - which turns out to be $log (1 - \hat{x}_k) $.


<h3>Real values:</h3>
$$L(W) = - \frac{1}{2}\sum_{k} (\hat{x}_k- x_k \ )^2$$


For data where the value (not category) is important to reproduce, we can use the sum of squared errors (SSE) for our Loss function. This function is usually used in regressions.

As it was with the above example, $x_k$ is one of our inputs and $\hat{x}_k$ is the respective output, and we want to make our output as similar as possible to our input.


<h3>Computing Gradient</h3>

The gradient of the loss function is an important and complex function.  It is defined as:
    $$\nabla_{W} L(W)_j = \frac{\partial f(x,W)}{\partial{W_j}}$$

Fortunately for us, TensorFlow computes these complex functions automatically when we define our functions that are used to compute loss!  They automatically manage the backpropagation algorithm, which is an efficient way of computing the gradients in complex neural networks.


<hr>


<a id="ref6"></a>

<h2>Code</h2>

 We are going to use the MNIST dataset for our example.
The following code was created by Aymeric Damien. You can find some of his code in <a href="https://github.com/aymericdamien">here</a>. We made some modifications which allow us to import the datasets to Jupyter Notebooks.


Let's call our imports and make the MNIST data available to use.



```python
#from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
%matplotlib inline

if not tf.__version__ == '2.9.0':
    print(tf.__version__)
    raise ValueError('please upgrade to TensorFlow 2.9.0, or restart your Kernel (Kernel->Restart & Clear Output)')


```


```python
import geopandas as gpd
gdf_perche = gpd.read_file("BIG_MO_RIVER.shp" )
```


```python
import numpy as np

# Convert the relevant columns to a NumPy array
z_values = gdf_perche[[f'z{i}' for i in range(100)]].values

# Calculate the first derivatives using vectorized operations
dx = (z_values[:, 1:] - z_values[:, :-1]) / 3

dx = dx[~np.isnan(dx).all(axis=1)]


dx.shape
```




    (244, 99)




```python
dx_norm = (dx-dx.min())/(dx.max() - dx.min())
dx_norm.shape
```




    (244, 99)




```python
# Calculate the second derivatives
d2x = (dx[:, 1:] - dx[:, :-1]) / 3
d2x.shape
```




    (244, 98)




```python
d2x_norm = (d2x-d2x.min())/(d2x.max() - d2x.min())
d2x_norm
```




    array([[0.82528414, 0.13099089, 0.71432308, ..., 0.53826379, 0.53924186,
            0.52107673],
           [0.59324414, 0.4968315 , 0.54046382, ..., 0.69552991, 0.52009345,
            0.56407951],
           [0.53919555, 0.57012852, 0.48839875, ..., 0.57604268, 0.55655119,
            0.53444936],
           ...,
           [0.52474613, 0.53402857, 0.54450567, ..., 0.48102924, 0.54663538,
            0.60615279],
           [0.54332495, 0.53606044, 0.52931677, ..., 0.54469194, 0.53304864,
            0.5280957 ],
           [0.53201523, 0.53775662, 0.52531005, ..., 0.4882252 , 0.63115479,
            0.48896523]])




```python
x = np.array(gdf_perche.iloc[:,6:-2])
print(x.shape)
# # Remove rows where all elements are NaN
x = x[~np.isnan(x).all(axis=1)]


x.shape
```

    (244, 100)
    




    (244, 100)




```python
x
```




    array([[191.5456789 , 191.74482882, 193.96124124, ..., 191.70522633,
            191.75559676, 191.70252287],
           [193.10962485, 192.77706825, 192.84416459, ..., 192.28942807,
            192.0521102 , 192.0111313 ],
           [190.40684509, 190.34706095, 190.31014336, ..., 191.9051269 ,
            192.00412284, 192.09289843],
           ...,
           [156.42172026, 156.47009645, 156.44060863, ..., 157.62332543,
            157.35138236, 157.56908167],
           [157.5691194 , 157.51177739, 157.50608906, ..., 156.99620052,
            157.02690268, 157.00309156],
           [156.53322672, 156.55081368, 156.54121135, ..., 161.18815975,
            161.51      , 161.50453852]])




```python
# Efficient vectorized approach
x = x - x.min(axis=1, keepdims=True)
x
```




    array([[0.41992905, 0.61907897, 2.8354914 , ..., 0.57947648, 0.62984691,
            0.57677302],
           [2.00930665, 1.67675005, 1.74384639, ..., 1.18910986, 0.951792  ,
            0.9108131 ],
           [0.09670174, 0.03691759, 0.        , ..., 1.59498354, 1.69397948,
            1.78275507],
           ...,
           [1.48958234, 1.53795853, 1.50847072, ..., 2.69118752, 2.41924444,
            2.63694375],
           [2.7937581 , 2.7364161 , 2.73072776, ..., 2.22083922, 2.25154139,
            2.22773026],
           [1.70516745, 1.7227544 , 1.71315207, ..., 6.36010047, 6.68194072,
            6.67647924]])




```python
max_x = x.max()
print(max_x)
```

    10.162187196255502
    


```python
x = x/ max_x
x
```




    array([[0.0413227 , 0.06091986, 0.27902373, ..., 0.05702281, 0.06197946,
            0.05675678],
           [0.19772384, 0.16499893, 0.17160148, ..., 0.11701318, 0.09366015,
            0.08962766],
           [0.00951584, 0.00363284, 0.        , ..., 0.15695278, 0.16669438,
            0.17543025],
           ...,
           [0.14658088, 0.15134129, 0.14843957, ..., 0.26482365, 0.23806336,
            0.25948585],
           [0.27491701, 0.26927432, 0.26871457, ..., 0.21853949, 0.22156071,
            0.2192176 ],
           [0.16779532, 0.16952595, 0.16858104, ..., 0.62585941, 0.65752978,
            0.65699235]])




```python
x = np.hstack([x, dx, d2x])
x.shape
```


```python
y = np.array(gdf_perche.iloc[:,-2])
y
```




    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1], dtype=int64)




```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=42)
```


```python
x_train
```




    array([[0.10399623, 0.10313987, 0.10376495, ..., 0.04301942, 0.04017536,
            0.03947518],
           [0.28797628, 0.29098566, 0.28968566, ..., 0.        , 0.01001222,
            0.01637824],
           [0.02279355, 0.02321638, 0.02718193, ..., 0.09365907, 0.09387346,
            0.08984763],
           ...,
           [0.08086448, 0.0367494 , 0.03430312, ..., 0.06654834, 0.05662498,
            0.05381599],
           [0.02251915, 0.02766366, 0.03790701, ..., 0.08757049, 0.09133154,
            0.09791686],
           [0.18115316, 0.05880894, 0.08027636, ..., 0.35069596, 0.35523   ,
            0.35662698]])



Now, let's give the parameters that are going to be used by our NN.



```python
learning_rate = 0.001
training_epochs = 150
batch_size = 10
display_step = 1
examples_to_show = 10
global_step = tf.Variable(0)
total_batch = int(len(x_train) / batch_size)

# Network Parameters
n_hidden_1 = 1000 # 1st layer num features
n_hidden_2 = 500 # 2nd layer num features
n_hidden_3 = 250 # 3rd layer num features





encoding_layer = 10 # final encoding bottleneck features
n_input = 100 # MNIST data input (img shape: 28*28)


```

<h3> encoder </h3>
Now we need to create our encoder. For this, we are going to use tf.keras.layers.Dense with sigmoidal activation functions. Sigmoidal functions delivers great results with this type of network. This is due to having a good derivative that is well-suited to backpropagation. We can create our encoder using the sigmoidal function like this:



```python

enocoding_1 = tf.keras.layers.Dense(n_hidden_1, activation=tf.nn.sigmoid)
encoding_2 = tf.keras.layers.Dense(n_hidden_2, activation=tf.nn.sigmoid)
encoding_3 = tf.keras.layers.Dense(n_hidden_3, activation=tf.nn.sigmoid)



encoding_final = tf.keras.layers.Dense(encoding_layer, activation=tf.nn.relu)

# Building the encoder
def encoder(x):
    x_reshaped = flatten_layer(x)
    # Encoder first layer with sigmoid activation #1
    layer_1 = enocoding_1(x_reshaped)
    # Encoder second layer with sigmoid activation #2
    layer_2 = encoding_2(layer_1)
    
    layer_3 = encoding_3(layer_2)
    

    code = encoding_final(layer_3)
    
    return code
```

<h3> decoder </h3>

You can see that the layer_1 in the encoder is the layer_2 in the decoder and vice-versa.



```python
decoding_1 = tf.keras.layers.Dense(n_hidden_3, activation=tf.nn.sigmoid)
decoding_2 = tf.keras.layers.Dense(n_hidden_2, activation=tf.nn.sigmoid)
decoding_3 = tf.keras.layers.Dense(n_hidden_1, activation=tf.nn.sigmoid)


decoding_final = tf.keras.layers.Dense(n_input)
# Building the decoder
def decoder(x):
    # Decoder first layer with sigmoid activation #1
    layer_1 = decoding_1(x)
    # Decoder second layer with sigmoid activation #2
    layer_2 = decoding_2(layer_1)
    
    layer_3 = decoding_3(layer_2)


    decode = self.decoding_final(layer_3)
    return decode
```

Let's construct our model.
We  define a <code>cost</code> function to calculate the loss  and a <code>grad</code> function to calculate gradients that will be used in backpropagation.



```python
class AutoEncoder(tf.keras.Model):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.n_hidden_1 = n_hidden_1 # 1st layer num features
        self.n_hidden_2 = n_hidden_2 # 2nd layer num features
        self.n_hidden_3 = n_hidden_3 # 2nd layer num features



        self.encoding_layer = encoding_layer
        self.n_input = n_input # MNIST data input (img shape: 28*28)

        self.flatten_layer = tf.keras.layers.Flatten()
        self.enocoding_1 = tf.keras.layers.Dense(self.n_hidden_1, activation=tf.nn.sigmoid)
        self.encoding_2 = tf.keras.layers.Dense(self.n_hidden_2, activation=tf.nn.sigmoid)
        self.encoding_3 = tf.keras.layers.Dense(self.n_hidden_3, activation=tf.nn.sigmoid)


        self.encoding_final = tf.keras.layers.Dense(self.encoding_layer, activation=tf.nn.relu)
        self.decoding_1 = tf.keras.layers.Dense(self.n_hidden_3, activation=tf.nn.sigmoid)
        self.decoding_2 = tf.keras.layers.Dense(self.n_hidden_2, activation=tf.nn.sigmoid)
        self.decoding_3 = tf.keras.layers.Dense(self.n_hidden_1, activation=tf.nn.sigmoid)


        self.decoding_final = tf.keras.layers.Dense(self.n_input)


    # Building the encoder
    def encoder(self,x):
        x = self.flatten_layer(x)
        layer_1 = self.enocoding_1(x)
        layer_2 = self.encoding_2(layer_1)
        layer_3 = self.encoding_3(layer_2)


        code = self.encoding_final(layer_3)
        return code
        

    # Building the decoder
    def decoder(self, x):
        layer_1 = self.decoding_1(x)
        layer_2 = self.decoding_2(layer_1)
        layer_3 = self.decoding_3(layer_2)


        decode = self.decoding_final(layer_3)
        return decode

        
    def call(self, x):
        encoder_op  = self.encoder(x)
        # Reconstructed Images
        y_pred = self.decoder(encoder_op)
        return y_pred
        
# def cost(y_true, y_pred):
#     loss = tf.losses.mean_squared_error(y_true, y_pred)
#     cost = tf.reduce_mean(loss)
#     return cost


def cost(y_true, y_pred):
    loss = tf.losses.mean_squared_error(y_true, y_pred)
    cost = tf.reduce_mean(loss)
    return cost



def grad(model, inputs, targets, loss_fn, optimizer): # use this with cross entropy to avois too large gradients
    with tf.GradientTape() as tape:
        reconstruction = model(inputs)
        loss_value = loss_fn(targets, reconstruction)
    
    gradients = tape.gradient(loss_value, model.trainable_variables)
    gradients, _ = tf.clip_by_global_norm(gradients, 1.0)  # Gradient clipping
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss_value, gradients, reconstruction


    
# def grad(model, inputs, targets, loss_fn, optimizer): # use this with cross entropy to avois too large gradients
#     with tf.GradientTape() as tape:
#         reconstruction = model(inputs)
#         loss_value = loss_fn(targets, reconstruction)
    
#     gradients = tape.gradient(loss_value, model.trainable_variables)
#     gradients, _ = tf.clip_by_global_norm(gradients, 1.0)  # Gradient clipping
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
#     return loss_value, gradients, reconstruction
```

For training we will run for 20 epochs.



```python
model = AutoEncoder()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

for epoch in range(training_epochs):
    for i in range(total_batch):
        x_inp = x_train[i : i + batch_size]
        loss_value, grads, reconstruction = grad(model, x_inp, x_inp, cost, optimizer)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch+1),
              "cost=", "{:.9f}".format(loss_value))

print("Optimization Finished!")
```

Now, let's apply encoder and decoder for our tests.



```python
encoded_values = model.encoder(tf.convert_to_tensor(x_test) )
```


```python
data = encoded_values.numpy()
data.shape
```


```python
y_s=y_test*255
```


```python
y_s
```


```python
data = encoded_values.numpy()
data.shape
```


```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

```


```python
tsne = TSNE(n_components=2, random_state=42)  # Set n_components=2 for 2D visualization
```


```python
downsampled_data = data_all.T[::2]
downsampled_data.shape
```


```python
tsne = TSNE(n_components=2, random_state=42)  # Set n_components=2 for 2D visualization
```


```python
data_tsne = tsne.fit_transform(data)
```


```python
data_tsne.shape
```


```python
data_all = np.array([data_tsne.T[0], data_tsne.T[1], y_s])
```


```python
data_all
```


```python
import altair as alt
import pandas as pd

# Assuming downsampled_data is already defined and contains the 'x' and 'y' columns
# Convert the data to a pandas DataFrame
df_plot = pd.DataFrame(data_tsne, columns=['x', 'y'])

# Create the scatter plot with x and y variables
scatter = alt.Chart(df_plot).mark_point(size=10, opacity = 0.2).encode(
    x='x:Q',
    y='y:Q'
).properties(
    title='Scatter Plot of x and y',
    width=600,  # Width of the plot
    height=600  # Height of the plot
)

# Display the plot
scatter

```


```python
encode_decode = model(tf.convert_to_tensor(x_test))
```


```python
all_ = list(range(3635))
import random
samples = random.sample(all_, 1000)
```


```python
all_ = list(range(3635))
import random
samples = random.sample(all_, 1000)

for i in samples:
    data1 = x_test[i][0:99]
    data2 = encode_decode[i][0:99].numpy()
    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Plot the first array
    axs[0].plot(data1, 'b-')
    axs[0].set_title('Actual')
    axs[0].set_xlabel('Distance in ft')
    axs[0].set_ylabel('Elevation (ft)')
    axs[0].set_ylim(0, 1)
    axs[0].text(0.5, 0.9, f'f0 = {data[i][0]}\n f1 = {data[i][1]}\n f2 = {data[i][2]}\n f3 = {data[i][3]}\n f4 = {data[i][4]}\n f5 = {data[i][5]}\n f6 = {data[i][6]}\nf7 = {data[i][7]}\n f8 = {data[i][8]}\n f9 = {data[i][9]}\n', 
                transform=axs[0].transAxes, fontsize=12, verticalalignment='top', 
                horizontalalignment='center', color='red')

    
    # Plot the second array
    axs[1].plot(data2, 'r-')
    axs[1].set_title('Generated')
    axs[1].set_xlabel('distance in ft')
    axs[1].set_ylim(0, 1)
    axs[1].text(0.5, 0.9, f't-SNE 1 = {data_tsne[i][0]}\n t-SNE 2 = {data_tsne[i][1]}', 
                transform=axs[1].transAxes, fontsize=12, verticalalignment='top', 
                horizontalalignment='center', color='red')

    # Display the plots
    plt.tight_layout()
    plt.show()
    
```


```python
model = ConvAutoEncoder()
optimizer = tf.keras.optimizers.Adam(learning_rate =0.001)

for epoch in range(training_epochs):
    for i in range(total_batch):
        x_inp = x_train[i : i + batch_size]
        loss_value, grads, reconstruction = grad(model, x_inp, x_inp, cost, optimizer)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch+1),
              "cost=", "{:.9f}".format(loss_value))

print("Optimization Finished!")
```


```python
import pickle

# Save the model
with open('AE_3_9_33_99_198_396.pkl', 'wb') as file:
    pickle.dump(model, file)
```


```python
# Load the model
with open('AE_3_9_33_99_198_396.pkl', 'rb') as file:
    model = pickle.load(file)
```


```python
flatten_layer(x_test).shape
```


```python
np.reshape()
```


```python
# Get the encoded representations for the test data
encoded_values = model.encoder(x_test)
y_s = y_test*255
```


```python
y_s
```


```python
data = encoded_values.numpy()
```


```python
# Plot the histogram
plt.hist(data.T[1], bins=200, edgecolor='black')

# Add labels and title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')

# Display the plot
plt.show()
```


```python
data = np.array([encoded_values.numpy().T[0], encoded_values.numpy().T[3], y_s])
```


```python
np.save('encoder.npy', encoded_values.numpy())
```


```python
# Applying encode and decode over test set
encode_decode = model(flatten_layer(x_image_test[:examples_to_show]))
```

Let's simply visualize our graphs!



```python
# Compare original images with their reconstructions
f, a = plt.subplots(2, 10, figsize=(15, 3))
for i in range(examples_to_show):
    a[0][i].imshow(np.reshape(x_image_test[i], (28, 28)))
    a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
```


```python
!pip install datashader
```


```python
import matplotlib.pyplot as plt

```


```python
import numpy as np
import pandas as pd
```


```python
data = np.load('encoder.npy')

```


```python
from holoviews import opts
from holoviews import hv
```


```python
!pip install --upgrade pyarrow
```


```python
!pip install "vegafusion[embed]>=1.5.0"
```


```python
import numpy as np
import pandas as pd
import altair as alt
```


```python
# Systematic sampling
downsampled_data = data.T[::3]
```


```python
downsampled_data.shape
```


```python
np.mean(data.T[2])
```


```python
downsampled_data
```


```python
import altair as alt
import pandas as pd

# Assuming downsampled_data is already defined and contains the 'x', 'y', and 'number' columns
# Convert the data to a pandas DataFrame
df = pd.DataFrame(downsampled_data, columns=['x', 'y', 'number'])

# Create a mapping from numbers to shapes
number_to_shape = {
    0: 'circle',   # Circle for 0
    1: 'square',   # Square for 1
    2: 'triangle', # Triangle for 2
    3: 'cross',    # Cross for 3
    4: 'diamond',  # Diamond for 4
    5: 'triangle-up',  # Triangle-up for 5
    6: 'triangle-down',# Triangle-down for 6
    7: 'triangle-right', # Triangle-right for 7
    8: 'triangle-left',  # Triangle-left for 8
    9: 'plus'       # Plus for 9
}

# Add a new column 'shape' based on 'number' column
df['shape'] = df['number'].map(number_to_shape)

# Create the scatter plot with custom shapes and a continuous color scale
scatter = alt.Chart(df).mark_point().encode(
    x='x:Q',
    y='y:Q',
    shape=alt.Shape('number:N', scale=alt.Scale(domain=[0,1,2,3,4,5,6,7,8,9, 10])),
    color=alt.Color('number:N', scale=alt.Scale(scheme='set1', domain=[0,1,2,3,4,5,6,7,8,9, 10])), # Change 'viridis' to your preferred scheme
    size=alt.value(20)  # Adjust the size as needed
).properties(
    title='Scatter Plot with Custom Size and Shapes',
    width=1200,  # Width of the plot
    height=1200  # Height of the plot
)

# Display the plot
scatter

```

As you can see, the reconstructions were successful. It can be seen that some noise were added to the image.



```python
from tensorflow.keras import layers, models

# Define the model to extract the latent layer
encoder = models.Model(inputs=model.input, outputs=model.get_layer('encoder_op').output)

# Get the latent representations of the input images
latent_representations = encoder.predict(x_test)
```

## Want to learn more?

Also, you can use **Watson Studio** to run these notebooks faster with bigger datasets.**Watson Studio** is IBM’s leading cloud solution for data scientists, built by data scientists. With Jupyter notebooks, RStudio, Apache Spark and popular libraries pre-packaged in the cloud, **Watson Studio** enables data scientists to collaborate on their projects without having to install anything. Join the fast-growing community of **Watson Studio** users today with a free account at [Watson Studio](https://cocl.us/ML0120EN_DSX).This is the end of this lesson. Thank you for reading this notebook, and good luck on your studies.


### Thanks for completing this lesson!


Created by <a href="https://www.linkedin.com/in/franciscomagioli?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0120ENSkillsNetwork954-2023-01-01">Francisco Magioli</a>, <a href="https://ca.linkedin.com/in/erich-natsubori-sato?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0120ENSkillsNetwork954-2023-01-01">Erich Natsubori Sato</a>, <a href="https://ca.linkedin.com/in/saeedaghabozorgi?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0120ENSkillsNetwork954-2023-01-01">Saeed Aghabozorgi</a>

Updated to TF 2.X by  <a href="https://www.linkedin.com/in/samaya-madhavan?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0120ENSkillsNetwork954-2023-01-01"> Samaya Madhavan </a>


### References:

-   [https://en.wikipedia.org/wiki/Autoencoder](https://en.wikipedia.org/wiki/Autoencoder?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0120ENSkillsNetwork954-2023-01-01&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ)
-   [http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/](http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0120ENSkillsNetwork954-2023-01-01&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ)
-   [http://www.slideshare.net/billlangjun/simple-introduction-to-autoencoder](http://www.slideshare.net/billlangjun/simple-introduction-to-autoencoder?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0120ENSkillsNetwork954-2023-01-01&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ)
-   [http://www.slideshare.net/danieljohnlewis/piotr-mirowski-review-autoencoders-deep-learning-ciuuk14](http://www.slideshare.net/danieljohnlewis/piotr-mirowski-review-autoencoders-deep-learning-ciuuk14?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0120ENSkillsNetwork954-2023-01-01&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ)
-   [https://cs.stanford.edu/~quocle/tutorial2.pdf](https://cs.stanford.edu/~quocle/tutorial2.pdf?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0120ENSkillsNetwork954-2023-01-01&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ)
-   <https://gist.github.com/hussius/1534135a419bb0b957b9>
-   [http://www.deeplearningbook.org/contents/autoencoders.html](http://www.deeplearningbook.org/contents/autoencoders.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0120ENSkillsNetwork954-2023-01-01&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ)
-   [http://www.kdnuggets.com/2015/03/deep-learning-curse-dimensionality-autoencoders.html/](http://www.kdnuggets.com/2015/03/deep-learning-curse-dimensionality-autoencoders.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0120ENSkillsNetwork954-2023-01-01&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ)
-   [https://www.youtube.com/watch?v=xTU79Zs4XKY](https://www.youtube.com/watch?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0120ENSkillsNetwork954-2023-01-01&v=xTU79Zs4XKY&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ)
-   [http://www-personal.umich.edu/~jizhu/jizhu/wuke/Stone-AoS82.pdf](http://www-personal.umich.edu/~jizhu/jizhu/wuke/Stone-AoS82.pdf?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0120ENSkillsNetwork954-2023-01-01&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ)


<hr>

Copyright © 2018 [Cognitive Class](https://cocl.us/DX0108EN_CC). This notebook and its source code are released under the terms of the [MIT License](https://bigdatauniversity.com/mit-license?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0120ENSkillsNetwork954-2023-01-01&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork-20629446&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ).


### Convolutional Layer


```python
import tensorflow as tf

# AutoEncoder class with 1D convolutional layers
class AutoEncoder(tf.keras.Model):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # Network Parameters
        self.n_hidden_1 = n_hidden_1  # Number of units in the first dense layer
        self.n_hidden_2 = n_hidden_2  # Number of units in the second dense layer
        self.encoding_layer = encoding_layer  # Bottleneck layer size
        self.n_input = n_input  # Input size (1D vector: 100)

        # 1D Convolutional layer for feature extraction
        self.conv_layer = tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', padding='same')

        # Flatten layer after convolution
        self.conv_flatten_layer = tf.keras.layers.Flatten()

        # Fully connected layers for encoding
        self.encoding_1 = tf.keras.layers.Dense(self.n_hidden_1, activation=tf.nn.sigmoid)
        self.encoding_2 = tf.keras.layers.Dense(self.n_hidden_2, activation=tf.nn.sigmoid)
        self.encoding_final = tf.keras.layers.Dense(self.encoding_layer, activation=tf.nn.relu)

        # Fully connected layers for decoding
        self.decoding_1 = tf.keras.layers.Dense(self.n_hidden_2, activation=tf.nn.sigmoid)
        self.decoding_2 = tf.keras.layers.Dense(self.n_hidden_1, activation=tf.nn.sigmoid)
        self.decoding_final = tf.keras.layers.Dense(self.n_input)

        # 1D Deconvolutional layer for reconstructing the input
        self.deconv_layer = tf.keras.layers.Conv1DTranspose(1, kernel_size=3, activation='sigmoid', padding='same')

    # Encoder function: maps input to the encoded representation
    def encoder(self, x):
        # Reshape input to 1D vector
        x = tf.reshape(x, (-1, 100, 1))
        x = self.conv_layer(x)
        x = self.conv_flatten_layer(x)
        x = self.encoding_1(x)
        x = self.encoding_2(x)
        code = self.encoding_final(x)
        return code

    # Decoder function: reconstructs the input from the encoded representation
    def decoder(self, x):
        x = self.decoding_1(x)
        x = self.decoding_2(x)
        x = self.decoding_final(x)
        # Reshape output to 1D vector
        x = tf.reshape(x, (-1, 100, 1))
        x = self.deconv_layer(x)
        # Flatten the output to match the original input shape
        decode = tf.reshape(x, (-1, self.n_input))
        return decode

    # Forward pass: encoding and then decoding the input
    def call(self, x):
        encoder_op = self.encoder(x)
        y_pred = self.decoder(encoder_op)
        return y_pred
    
def cost(y_true, y_pred):
    # Base reconstruction loss
#     y_pred = tf.squeeze(y_pred, axis=-1)
#     y_true = tf.squeeze(y_true, axis=-1)
    loss1 = tf.abs(y_true - y_pred)
    reconstruction_loss = tf.reduce_mean(loss1)
    
    grad_true = y_true[:, 1:] - y_true[:, :-1]
    grad_true = tf.pad(grad_true, [[0, 0], [0, 1]], "CONSTANT")

    grad_pred = y_pred[:, 1:] - y_pred[:, :-1]
    grad_pred = tf.pad(grad_pred, [[0, 0], [0, 1]], "CONSTANT")
    
    grad_true = tf.cast(grad_true, dtype=tf.float32)
    grad_pred = tf.cast(grad_pred, dtype=tf.float32)

    # Penalize if the gradient of the prediction is smaller than the gradient of the true data (indicating smoothing)
    loss2 = tf.abs(grad_true - grad_pred)
    penalty = tf.reduce_mean(loss2)

    #############################################
    grad2_true = grad_true[:, 1:] - grad_true[:, :-1]
    grad2_true = tf.pad(grad2_true, [[0, 0], [0, 1]], "CONSTANT")

    grad2_pred = grad_pred[:, 1:] - grad_pred[:, :-1]
    grad2_pred = tf.pad(grad2_pred, [[0, 0], [0, 1]], "CONSTANT")
    
    grad2_true = tf.cast(grad2_true, dtype=tf.float32)
    grad2_pred = tf.cast(grad2_pred, dtype=tf.float32)

    # Penalize if the gradient of the prediction is smaller than the gradient of the true data (indicating smoothing)
    loss3 = tf.abs(grad2_true - grad2_pred)
    penalty2 = tf.reduce_mean(loss3)
    ######################################################
    # Combine with a weighting factor
    cost = reconstruction_loss + 100 * penalty + 100 * penalty2  # Adjust the weight as needed
    loss = loss1 +loss2
    return cost



def grad(model, inputs, targets, loss_fn, optimizer): # use this with cross entropy to avois too large gradients
    with tf.GradientTape() as tape:
        reconstruction = model(inputs)
        loss_value = loss_fn(targets, reconstruction)
    
    gradients = tape.gradient(loss_value, model.trainable_variables)
    gradients, _ = tf.clip_by_global_norm(gradients, 1.0)  # Gradient clipping
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss_value, gradients, reconstruction
```


```python
# Training loop with gradient clipping
model = AutoEncoder()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

for epoch in range(training_epochs):
    for i in range(total_batch):
        x_inp = x_train[i * batch_size: (i + 1) * batch_size]
        loss_value, grads, reconstruction = grad(model, x_inp, x_inp, cost, optimizer)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(loss_value))

print("Optimization Finished!")
```


```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import altair as alt
import pandas as pd


encoded_values = model.encoder(tf.convert_to_tensor(x))
data = encoded_values.numpy()
print(data.shape)

tsne = TSNE(n_components=2, random_state=42)  # Set n_components=2 for 2D visualization
```


```python
data_tsne = tsne.fit_transform(data)
print(data_tsne.shape)
```


```python
y_s=y
```


```python
data_all = np.array([data_tsne.T[0], data_tsne.T[1], y_s])
```


```python
# data_all.shape
```


```python
downsampled_data_ = data_all.T[::4]
downsampled_data_.shape
```


```python
import altair as alt
import pandas as pd

# Assuming downsampled_data is already defined and contains the 'x', 'y', and 'number' columns
# Convert the data to a pandas DataFrame
df = pd.DataFrame(data_all.T, columns=['x', 'y', 'number'])

# Create a mapping from numbers to shapes
number_to_shape = {
    0: 'not defined',   # Circle for 0
    1: 'levees',   # Square for 1
    2: 'channels', # Triangle for 2
    3: 'cross',    # Cross for 3
    4: 'diamond',  # Diamond for 4
    5: 'triangle-up',  # Triangle-up for 5
    6: 'triangle-down',# Triangle-down for 6
    7: 'triangle-right', # Triangle-right for 7
    8: 'triangle-left',  # Triangle-left for 8
    9: 'plus'       # Plus for 9
}

# Add a new column 'shape' based on 'number' column
df['shape'] = df['number'].map(number_to_shape)

# Create the scatter plot with custom shapes and a continuous color scale
scatter = alt.Chart(df).mark_point().encode(
    x='x:Q',
    y='y:Q',
    shape=alt.Shape('number:N', scale=alt.Scale(domain=[8,2,9,3,4,5,6,7,0,1, 10])),
    color=alt.Color('number:N', scale=alt.Scale(scheme='set1', domain=[8,2,9,3,4,5,6,7,0,1, 10])), # Change 'viridis' to your preferred scheme
    size=alt.value(20)  # Adjust the size as needed
).properties(
    title='Scatter Plot with Custom Size and Shapes',
    width=800,  # Width of the plot
    height=800  # Height of the plot
)

# Display the plot
scatter

```


```python
x_test.shape
```


```python
encode_decode = model(tf.convert_to_tensor(x))

all_ = list(range(244))
import random

samples = random.sample(all_, 100)

for i in all_:
    data1 =x[i]
    data2 = encode_decode[i].numpy()
    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    print(data1)
    # Plot the first array
    axs[0].plot(data1, 'b-')
    axs[0].set_title('Actual')
    axs[0].set_xlabel('Distance in yard')
    axs[0].set_ylabel('Elevation (ft)')
#     axs[0].set_ylim(0, 2.5)
    axs[0].text(0.5, 0.9, f'sigma1 = {data[i][0]}\n sigma2 = {data[i][1]}\n H1 = {data[i][2]}\n H2 = {data[i][3]}\n d1 = {data[i][4]}\n d2 = {data[i][5]}\n t = {data[i][6]}\n R = {data[i][7]*(180/np.pi)} degree\n', 
                transform=axs[0].transAxes, fontsize=12, verticalalignment='top', 
                horizontalalignment='center', color='red')

    
    # Plot the second array
    axs[1].plot(data2, 'r-')
    axs[1].set_title('Generated')
    axs[1].set_xlabel('distance in ft')
    axs[1].set_ylim(0, 0.20)
    axs[1].set_xlim(-100, 110)

    axs[1].text(0.5, 0.9, f't-SNE 1 = {data_tsne[i][0]}\n t-SNE 2 = {data_tsne[i][1]}', 
                transform=axs[1].transAxes, fontsize=12, verticalalignment='top', 
                horizontalalignment='center', color='red')

    # Display the plots
    plt.tight_layout()
    plt.show()
    
```


```python
import altair as alt
import pandas as pd

# Assuming downsampled_data is already defined and contains the 'x' and 'y' columns
# Convert the data to a pandas DataFrame
df_plot = pd.DataFrame(data_tsne, columns=['x', 'y'])

# Create the scatter plot with x and y variables
scatter = alt.Chart(df_plot).mark_point(size=10, opacity = 0.2).encode(
    x='x:Q',
    y='y:Q'
).properties(
    title='Scatter Plot of x and y',
    width=600,  # Width of the plot
    height=600  # Height of the plot
)

# Display the plot
scatter

```

## Guided AutoEncoder


```python
data.T[3]
```


```python
learning_rate = 0.001
training_epochs = 150
batch_size = 10
display_step = 1
examples_to_show = 10
global_step = tf.Variable(0)
total_batch = int(len(x_train) / batch_size)

# Network Parameters
n_hidden_1 = 1000 # 1st layer num features
n_hidden_2 = 500 # 2nd layer num features
n_hidden_3 = 250 # 3rd layer num features




encoding_layer = 8 # final encoding bottleneck features
n_input = 100 # MNIST data input (img shape: 28*28)
```


```python
import tensorflow as tf

# AutoEncoder class with 1D convolutional layers
class AutoEncoder_guided(tf.keras.Model):
    def __init__(self):
        super(AutoEncoder_guided, self).__init__()

        # Network Parameters
        self.n_hidden_1 = n_hidden_1  # Number of units in the first dense layer
        self.n_hidden_2 = n_hidden_2  # Number of units in the second dense layer
        self.encoding_layer = encoding_layer  # Bottleneck layer size
        self.n_input = n_input  # Input size (1D vector: 100)

        # 1D Convolutional layer for feature extraction
        self.conv_layer = tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', padding='same')

        # Flatten layer after convolution
        self.conv_flatten_layer = tf.keras.layers.Flatten()

        # Fully connected layers for encoding
        self.encoding_1 = tf.keras.layers.Dense(self.n_hidden_1, activation=tf.nn.sigmoid,
                                                kernel_initializer=tf.keras.initializers.HeNormal(), 
                                                bias_initializer=tf.keras.initializers.Ones())
        
        self.encoding_2 = tf.keras.layers.Dense(self.n_hidden_2, activation=tf.nn.sigmoid, 
                                                kernel_initializer=tf.keras.initializers.HeNormal(), 
                                                bias_initializer=tf.keras.initializers.Ones())
        
        self.encoding_final = tf.keras.Sequential([
           tf.keras.layers.Dense(self.encoding_layer, activation=tf.nn.relu),
           tf.keras.layers.Lambda(lambda x: x + 0.1)])
        # self.encoding_final = tf.keras.layers.Dense(self.encoding_layer, activation=tf.nn.relu)
        self.encoding_final = tf.keras.layers.Dense(self.encoding_layer, activation=tf.keras.activations.softmax)

        

        # Fully connected layers for decoding
        self.decoding_1 = tf.keras.layers.Dense(self.n_hidden_2, activation=tf.nn.sigmoid)
        self.decoding_2 = tf.keras.layers.Dense(self.n_hidden_1, activation=tf.nn.sigmoid)
        self.decoding_final = tf.keras.layers.Dense(self.n_input)

        # 1D Deconvolutional layer for reconstructing the input
        self.deconv_layer = tf.keras.layers.Conv1DTranspose(1, kernel_size=3, activation='sigmoid', padding='same')

        
    
    
    # Encoder function: maps input to the encoded representation
    def encoder(self, x):
        # Reshape input to 1D vector
        x = tf.reshape(x, (-1, 100, 1))
        x = self.conv_layer(x)
        x = self.conv_flatten_layer(x)
        x = self.encoding_1(x)
        x = self.encoding_2(x)
        x = self.encoding_final(x)
        return x

    # Decoder function: reconstructs the input from the encoded representation
    def decoder(self, x):
        def wavlet(x):
            rows = []
            for item in (x):
                # Extract parameters from x
                sigma1, sigma2, h1, h2, d1, d2, t, theta = tf.split(item, 8, axis = -1)
                # sigma1, h1, d1, t, theta = tf.split(item, 5, axis = -1)



                # Convert parameters to TensorFlow tensors
#                 sigma1 = 20.0
#                 sigma2 = tf.maximum(sigma2, 0.1)
#                 h1 = 5.0
#                 h2 = tf.maximum(h1, 1.0)
#                 d1 = tf.maximum(d2, 1.0)
#                 d2 = tf.maximum(d2, 1.0)

                # Generate x values
                x_ = tf.linspace(0.0, 99.0, 100)

                # Compute wavelet functions
                C1 = 2 / (tf.sqrt(3 * sigma1*100) * tf.sqrt(np.pi))
                C2 = 2 / (tf.sqrt(3 * sigma2*100) * tf.sqrt(np.pi))

                y1 = 100.*h1 * C1 * (1 - ((x_ + d1*100.)**2) / (sigma1*100.)**2) * tf.exp(-(x_ + d1*100.)**2 / (2 * (sigma1)**2))
                y2 = 100.*h2 * C2 * (1 - ((x_ + d2*100.)**2) / (sigma2*100.)**2) * tf.exp(-(x_ + d2*100.)**2 / (2 * (sigma2)**2))

                y = y1 + t  + y2
                                
                theta = theta[0]  # Assumes theta is a tensor of shape (1, )
                
                
                rotation_matrix = tf.convert_to_tensor([
                    [tf.cos(theta), -tf.sin(theta)],
                    [tf.sin(theta), tf.cos(theta)]
                ])


                # Stack x and y values
                coordinates = tf.stack([x_, y])

                # Rotate coordinates
                rotated_coordinates = tf.matmul(rotation_matrix, coordinates)

                # Extract rotated coordinates
                y_rotated = rotated_coordinates[1]

                # Append the rotated y values as a new row
                rows.append(y_rotated)

            tensor = tf.stack(rows)
            return tensor
        
        x = wavlet(x)
        x = tf.expand_dims(x, axis=-1)  # Add channel dimension
        x = self.deconv_layer(x)
        return x

    # Forward pass: encoding and then decoding the input
    def call(self, x):
        encoder_op = self.encoder(x)
        y_pred = self.decoder(encoder_op)
        return y_pred
    
def cost(y_true, y_pred):
    # Base reconstruction loss
    y_pred = tf.squeeze(y_pred, axis=-1)
#     y_true = tf.squeeze(y_true, axis=-1)
    loss1 = tf.abs(y_true - y_pred)
    reconstruction_loss = tf.reduce_mean(loss1)
    
    grad_true = y_true[:, 1:] - y_true[:, :-1]
    grad_true = tf.pad(grad_true, [[0, 0], [0, 1]], "CONSTANT")

    grad_pred = y_pred[:, 1:] - y_pred[:, :-1]
    grad_pred = tf.pad(grad_pred, [[0, 0], [0, 1]], "CONSTANT")
    
    grad_true = tf.cast(grad_true, dtype=tf.float32)
    grad_pred = tf.cast(grad_pred, dtype=tf.float32)

    # Penalize if the gradient of the prediction is smaller than the gradient of the true data (indicating smoothing)
    loss2 = tf.abs(grad_true - grad_pred)
    penalty = tf.reduce_mean(loss2)

    #############################################
    grad2_true = grad_true[:, 1:] - grad_true[:, :-1]
    grad2_true = tf.pad(grad2_true, [[0, 0], [0, 1]], "CONSTANT")

    grad2_pred = grad_pred[:, 1:] - grad_pred[:, :-1]
    grad2_pred = tf.pad(grad2_pred, [[0, 0], [0, 1]], "CONSTANT")
    
    grad2_true = tf.cast(grad2_true, dtype=tf.float32)
    grad2_pred = tf.cast(grad2_pred, dtype=tf.float32)

    # Penalize if the gradient of the prediction is smaller than the gradient of the true data (indicating smoothing)
    loss3 = tf.abs(grad2_true - grad2_pred)
    penalty2 = tf.reduce_mean(loss3)
    ######################################################
    # Combine with a weighting factor
    cost = reconstruction_loss + 10 * penalty + 10 * penalty2  # Adjust the weight as needed
    loss = loss1 +loss2
    return cost


# def cost(y_true, y_pred):
#     y_true = tf.expand_dims(y_true, axis=-1)
# #     loss = tf.losses.mean_squared_error(y_true, y_pred)
#     y_true = tf.cast(y_true, tf.float32)
#     y_pred = tf.cast(y_pred, tf.float32)
#     loss = tf.abs(y_true - y_pred)
#     cost = tf.reduce_mean(loss)
#     return cost    
    
def grad(model, inputs, targets, loss_fn, optimizer): # use this with cross entropy to avois too large gradients
    with tf.GradientTape() as tape:
        reconstruction = model(inputs)
        loss_value = loss_fn(targets, reconstruction)
    
    gradients = tape.gradient(loss_value, model.trainable_variables)
    gradients, _ = tf.clip_by_global_norm(gradients, 1.0)  # Gradient clipping
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss_value, gradients, reconstruction
```


```python
model = AutoEncoder_guided()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)


for epoch in range(training_epochs):
    for i in range(total_batch):
        x_inp = x_train[i : i + batch_size]
        loss_value, grads, reconstruction = grad(model, x_inp, x_inp, cost, optimizer)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch+1),
              "cost=", "{:.9f}".format(loss_value))

print("Optimization Finished!")
```

    Epoch: 0001 cost= 0.557679892
    Epoch: 0002 cost= 0.544078827
    Epoch: 0003 cost= 0.533094287
    Epoch: 0004 cost= 0.524866283
    Epoch: 0005 cost= 0.519019067
    Epoch: 0006 cost= 0.514452219
    Epoch: 0007 cost= 0.510735333
    Epoch: 0008 cost= 0.507483006
    Epoch: 0009 cost= 0.504675746
    Epoch: 0010 cost= 0.502161920
    Epoch: 0011 cost= 0.500092864
    Epoch: 0012 cost= 0.498322189
    Epoch: 0013 cost= 0.496818095
    Epoch: 0014 cost= 0.495464057
    Epoch: 0015 cost= 0.494249135
    Epoch: 0016 cost= 0.493135929
    Epoch: 0017 cost= 0.492100656
    Epoch: 0018 cost= 0.491109669
    Epoch: 0019 cost= 0.490163088
    Epoch: 0020 cost= 0.489297986
    Epoch: 0021 cost= 0.488548517
    Epoch: 0022 cost= 0.487806439
    Epoch: 0023 cost= 0.487139344
    Epoch: 0024 cost= 0.486480325
    Epoch: 0025 cost= 0.485904753
    Epoch: 0026 cost= 0.485322416
    Epoch: 0027 cost= 0.484784275
    Epoch: 0028 cost= 0.484353006
    Epoch: 0029 cost= 0.483950734
    Epoch: 0030 cost= 0.483563751
    Epoch: 0031 cost= 0.483199179
    Epoch: 0032 cost= 0.482850164
    Epoch: 0033 cost= 0.482512355
    Epoch: 0034 cost= 0.482161701
    Epoch: 0035 cost= 0.481795669
    Epoch: 0036 cost= 0.481444955
    Epoch: 0037 cost= 0.481133372
    Epoch: 0038 cost= 0.480849296
    Epoch: 0039 cost= 0.480591357
    Epoch: 0040 cost= 0.480366886
    Epoch: 0041 cost= 0.480201542
    Epoch: 0042 cost= 0.480062813
    Epoch: 0043 cost= 0.479952991
    Epoch: 0044 cost= 0.479847550
    Epoch: 0045 cost= 0.479750752
    Epoch: 0046 cost= 0.479649127
    Epoch: 0047 cost= 0.479597628
    Epoch: 0048 cost= 0.479553461
    Epoch: 0049 cost= 0.479508102
    Epoch: 0050 cost= 0.479439735
    Epoch: 0051 cost= 0.479405999
    Epoch: 0052 cost= 0.479374141
    Epoch: 0053 cost= 0.479316235
    Epoch: 0054 cost= 0.479270518
    Epoch: 0055 cost= 0.479256332
    Epoch: 0056 cost= 0.479237705
    Epoch: 0057 cost= 0.479219347
    Epoch: 0058 cost= 0.479202658
    Epoch: 0059 cost= 0.479188621
    Epoch: 0060 cost= 0.479176372
    Epoch: 0061 cost= 0.479159921
    Epoch: 0062 cost= 0.479151428
    Epoch: 0063 cost= 0.479138672
    Epoch: 0064 cost= 0.479137927
    Epoch: 0065 cost= 0.479133248
    Epoch: 0066 cost= 0.479130119
    Epoch: 0067 cost= 0.479128271
    Epoch: 0068 cost= 0.479128361
    Epoch: 0069 cost= 0.479125321
    Epoch: 0070 cost= 0.479122818
    Epoch: 0071 cost= 0.479121745
    Epoch: 0072 cost= 0.479120523
    Epoch: 0073 cost= 0.479115427
    Epoch: 0074 cost= 0.479114264
    Epoch: 0075 cost= 0.479103714
    Epoch: 0076 cost= 0.479108393
    Epoch: 0077 cost= 0.479109347
    Epoch: 0078 cost= 0.479109347
    Epoch: 0079 cost= 0.479108870
    Epoch: 0080 cost= 0.479104847
    Epoch: 0081 cost= 0.479105651
    Epoch: 0082 cost= 0.479102254
    Epoch: 0083 cost= 0.479100317
    Epoch: 0084 cost= 0.479097903
    Epoch: 0085 cost= 0.479096591
    Epoch: 0086 cost= 0.479092896
    Epoch: 0087 cost= 0.479093492
    Epoch: 0088 cost= 0.479093134
    Epoch: 0089 cost= 0.479089677
    Epoch: 0090 cost= 0.479091018
    Epoch: 0091 cost= 0.479089081
    Epoch: 0092 cost= 0.479081392
    Epoch: 0093 cost= 0.479090840
    Epoch: 0094 cost= 0.479083091
    Epoch: 0095 cost= 0.479087710
    Epoch: 0096 cost= 0.479072750
    Epoch: 0097 cost= 0.479086429
    Epoch: 0098 cost= 0.479081810
    Epoch: 0099 cost= 0.479082882
    Epoch: 0100 cost= 0.479071498
    Epoch: 0101 cost= 0.479081869
    Epoch: 0102 cost= 0.479081750
    Epoch: 0103 cost= 0.479081988
    Epoch: 0104 cost= 0.479082465
    Epoch: 0105 cost= 0.479066312
    Epoch: 0106 cost= 0.479078948
    Epoch: 0107 cost= 0.479078293
    Epoch: 0108 cost= 0.479081810
    Epoch: 0109 cost= 0.479066014
    Epoch: 0110 cost= 0.479078114
    Epoch: 0111 cost= 0.479075491
    Epoch: 0112 cost= 0.479077876
    Epoch: 0113 cost= 0.479064435
    Epoch: 0114 cost= 0.479079008
    Epoch: 0115 cost= 0.479075074
    Epoch: 0116 cost= 0.479075879
    Epoch: 0117 cost= 0.479062200
    Epoch: 0118 cost= 0.479076713
    Epoch: 0119 cost= 0.479073197
    Epoch: 0120 cost= 0.479071677
    Epoch: 0121 cost= 0.479060233
    Epoch: 0122 cost= 0.479072094
    Epoch: 0123 cost= 0.479072034
    Epoch: 0124 cost= 0.479058295
    Epoch: 0125 cost= 0.479070961
    Epoch: 0126 cost= 0.479073673
    Epoch: 0127 cost= 0.479053944
    Epoch: 0128 cost= 0.479068637
    Epoch: 0129 cost= 0.479074657
    Epoch: 0130 cost= 0.479053080
    Epoch: 0131 cost= 0.479069412
    Epoch: 0132 cost= 0.479076743
    Epoch: 0133 cost= 0.479057431
    Epoch: 0134 cost= 0.479070097
    Epoch: 0135 cost= 0.479068309
    Epoch: 0136 cost= 0.479057193
    Epoch: 0137 cost= 0.479068905
    Epoch: 0138 cost= 0.479069501
    Epoch: 0139 cost= 0.479068369
    Epoch: 0140 cost= 0.479057282
    Epoch: 0141 cost= 0.479070574
    Epoch: 0142 cost= 0.479067385
    Epoch: 0143 cost= 0.479056180
    Epoch: 0144 cost= 0.479069471
    Epoch: 0145 cost= 0.479068696
    Epoch: 0146 cost= 0.479067415
    Epoch: 0147 cost= 0.479055583
    Epoch: 0148 cost= 0.479068875
    Epoch: 0149 cost= 0.479067117
    Epoch: 0150 cost= 0.479065239
    Optimization Finished!
    

### Hinton Deep Belief Net and Auto Encoders


```python
import numpy as np
import tensorflow as tf
```


```python
input_size = 100
numvis = input_size
numhid = 1000
numpen = 500
numtop = 250
numlab = 8
```


```python
# Initialize weights
vishid = tf.Variable(tf.random.normal([input_size, numhid], stddev=0.01))
hidpen = tf.Variable(tf.random.normal([numhid, numpen], stddev=0.01))
pentop = tf.Variable(tf.random.normal([numpen, numtop], stddev=0.01))
labtop = tf.Variable(tf.random.normal([numlab, numtop], stddev=0.01))
penhid = tf.Variable(tf.random.normal([numpen, numhid], stddev=0.01))
hidvis = tf.Variable(tf.random.normal([numpen, numvis], stddev=0.01))

# Initialize biases
topbiases = tf.Variable(tf.zeros([1, numtop], dtype=tf.float32))
hidrecbiases = tf.Variable(tf.zeros([1, numhid], dtype=tf.float32))
penrecbiases = tf.Variable(tf.zeros([1, numpen], dtype=tf.float32))
hidgenbiases = tf.Variable(tf.zeros([1, numhid], dtype=tf.float32))
labgenbiases = tf.Variable(tf.zeros([1, numlab], dtype=tf.float32))
pengenbiases = tf.Variable(tf.zeros([1, numpen], dtype=tf.float32))
visgenbiases = tf.Variable(tf.zeros([1, numvis], dtype=tf.float32))
```


```python
# Logistic sigmoid function
def logistic(x):
    return 1.0 / (1.0 + tf.exp(-x))

# Softmax function
def softmax(x):
    exp_x = tf.exp(x - tf.reduce_max(x, axis=1, keepdims=True))
    return exp_x / tf.reduce_sum(exp_x, axis=1, keepdims=True)

# error function
def error(x, x1):
    return tf.reduce_mean(tf.square(x - x1))
```


```python
def wavlet(x):
        rows = []
        for item in (x):
            # Extract parameters from x
            sigma1, sigma2, h1, h2, d1, d2, t, theta = tf.split(item, 8, axis = -1)
            # sigma1, h1, d1, t, theta = tf.split(item, 5, axis = -1)



            # Convert parameters to TensorFlow tensors
#                 sigma1 = 20.0
#                 sigma2 = tf.maximum(sigma2, 0.1)
#                 h1 = 5.0
#                 h2 = tf.maximum(h1, 1.0)
#                 d1 = tf.maximum(d2, 1.0)
#                 d2 = tf.maximum(d2, 1.0)

            # Generate x values
            x_ = tf.linspace(0.0, 99.0, 100)

            # Compute wavelet functions
            C1 = 2 / (tf.sqrt(3 * sigma1*100) * tf.sqrt(np.pi))
            C2 = 2 / (tf.sqrt(3 * sigma2*100) * tf.sqrt(np.pi))

            y1 = 100.*h1 * C1 * (1 - ((x_ + d1*100.)**2) / (sigma1*100.)**2) * tf.exp(-(x_ + d1*100.)**2 / (2 * (sigma1)**2))
            y2 = 100.*h2 * C2 * (1 - ((x_ + d2*100.)**2) / (sigma2*100.)**2) * tf.exp(-(x_ + d2*100.)**2 / (2 * (sigma2)**2))

            y = y1 + t  + y2

            theta = theta[0]  # Assumes theta is a tensor of shape (1, )


            rotation_matrix = tf.convert_to_tensor([
                [tf.cos(theta), -tf.sin(theta)],
                [tf.sin(theta), tf.cos(theta)]
            ])


            # Stack x and y values
            coordinates = tf.stack([x_, y])

            # Rotate coordinates
            rotated_coordinates = tf.matmul(rotation_matrix, coordinates)

            # Extract rotated coordinates
            y_rotated = rotated_coordinates[1]

            # Append the rotated y values as a new row
            rows.append(y_rotated)

```


```python
batchsize =200
numCDiters = 10  # Number of Contrasive Diversion iterations
r = 0.01  # learning rate

#creating datasets
train_ds = \
    tf.data.Dataset.from_tensor_slices((trX[0:1000], trY_[0:1000])).batch(batchsize)


batch_number = 0
epochs = 10
for epoch in range(epochs):
    for data, targets in train_ds:
        batch_number +=1
        for i_sample in range(batchsize):
            # Bottom-up pass for wake/positive phase (Data-Driven)
            vis = data[i_sample]
            wakehidprobs = logistic(tf.matmul([vis], vishid) + hidrecbiases)
            wakehidstates = tf.cast(wakehidprobs > tf.random.uniform(tf.shape(wakehidprobs)), tf.float32)
            wakepenprobs = logistic(tf.matmul(wakehidstates, hidpen) + penrecbiases)
            wakepenstates = tf.cast(wakepenprobs > tf.random.uniform(tf.shape(wakepenprobs)), tf.float32)
            waketopprobs = logistic(tf.matmul(wakepenstates, pentop) + tf.matmul([targets[i_sample]], labtop) + topbiases)
            waketopstates = tf.cast(waketopprobs > tf.random.uniform(tf.shape(waketopprobs)), tf.float32)


            # Positive phase statistics
            poslabtopstatistics = tf.matmul(tf.transpose([targets[i_sample]]), waketopstates)
            pospentopstatistics = tf.matmul(tf.transpose(wakepenstates), waketopstates)

            # Gibbs sampling iterations for negative phase
            negtopstates = waketopstates
            for iter in range(numCDiters):
                negpenprobs = logistic(tf.matmul(negtopstates, tf.transpose(pentop)) + pengenbiases)
                negpenstates = tf.cast(negpenprobs > tf.random.uniform(tf.shape(negpenprobs)), tf.float32)
                neglabprobs = softmax(tf.matmul(negtopstates, tf.transpose(labtop)) + labgenbiases)
                negtopprobs = logistic(tf.matmul(negpenstates, pentop) + tf.matmul(neglabprobs, labtop) + topbiases)
                negtopstates = tf.cast(negtopprobs > tf.random.uniform(tf.shape(negtopprobs)), tf.float32)

            # Negative phase statistics for Contrasive Divergance
            negpentopstatistics = tf.matmul(tf.transpose(negpenstates), negtopstates)
            neglabtopstatistics = tf.matmul(tf.transpose(neglabprobs), negtopstates)


            # Top-down generative pass for sleep/negative phase
            sleeppenstates = negpenstates
            sleephidprobs = logistic(tf.matmul(sleeppenstates, penhid) + hidgenbiases)
            sleephidstates = tf.cast(sleephidprobs > tf.random.uniform(tf.shape(sleephidprobs)), tf.float32)
            sleepvisprobs = logistic(tf.matmul(sleephidstates, hidvis) + visgenbiases)
        
            # Predictions
            psleeppenstates = logistic(tf.matmul(sleephidstates, hidpen) + penrecbiases)
            psleephidstates = logistic(tf.matmul(sleepvisprobs, vishid) + hidrecbiases)
            pvisprobs = logistic(tf.matmul(wakehidstates, hidvis) + visgenbiases)
            phidprobs = logistic(tf.matmul(wakepenstates, penhid) + hidgenbiases)

            # Update generative parameters     
            hidvis = hidvis + (r * tf.matmul(tf.transpose(psleephidstates), [data[i_sample]] - pvisprobs))
            visgenbiases = visgenbiases + (r * ([data[i_sample]] - pvisprobs))
            penhid = penhid + (r * tf.matmul(tf.transpose(wakepenstates), wakehidstates - phidprobs))
            hidgenbiases = hidgenbiases + (r * (wakehidstates - phidprobs))


            # Update top level associative memory parameters
            labtop = labtop + (r * (poslabtopstatistics - neglabtopstatistics))
            labgenbiases = labgenbiases + (r * ([targets[i_sample]] - neglabprobs))
            pentop = pentop + (r * (pospentopstatistics - negpentopstatistics))
            pengenbiases = pengenbiases + (r * (wakepenstates - negpenstates))
            topbiases = topbiases + (r * (waketopstates - negtopstates))

            # Update recognition/inference approximation parameters
            hidpen = hidpen + (r * tf.matmul(tf.transpose(sleephidstates), sleeppenstates - psleeppenstates))
            penrecbiases = penrecbiases + (r * (sleeppenstates - psleeppenstates))
            vishid = vishid + (r * tf.matmul(tf.transpose(sleepvisprobs), sleephidstates - psleephidstates))
            hidrecbiases = hidrecbiases + (r * (sleephidstates - psleephidstates))

            if i_sample == batchsize-1:
                err = error([vis],pvisprobs)*100
                print ( 'Epoch: %d' % epoch, 
                           "batch #: %i " % batch_number, "of %i" % int(60e3/batchsize), 
                           "sample #: %i" % i_sample,
                           'error: %.2f ' % err)

```

# MSDIS LiDAR Downloader

A Python package to download and visualize LiDAR data from Missouri Spatial Data Information Services (MSDIS).

## Features
- Download LiDAR tiles based on a given shapefile.
- Visualize the LiDAR data tiles.
- Command-line interface for ease of use.

## Installation
You can install the package using pip:

```bash
pip install msdis-lidar
