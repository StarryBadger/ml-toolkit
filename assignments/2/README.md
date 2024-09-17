# Assignment 2 Report

To install dependencies, use 
```bash
pip install -r requirements.txt
```

To run any python3 file, run it from the base directory using,
```bash
python3 -m path.to.code.file # without .py extension
```
To run the code for this assignment,
```bash
python3 -m assignments.2.a2 # (without .py extension)
```
Website references are mentioned in the code.

# K-Means Clustering

## Elbow Plot for seed 43

![Elbow Plot](./figures/elbow_kmeans_8.png)

Optimal number of clusters (k_kmeans1): 8
Cluster labels: [4 4 6 2 2 7 4 4 7 6 2 4 2 1 1 0 1 4 4 3 4 4 6 4 6 0 6 4 7 0 6 5 2 5 2 7 7
 1 4 6 7 1 7 4 0 0 4 4 4 4 4 4 5 4 0 0 0 4 2 7 2 0 6 4 0 7 5 2 7 4 2 4 4 7
 5 2 0 1 0 3 4 0 0 2 1 4 7 0 0 1 3 2 4 6 4 1 7 1 0 4 1 1 0 4 7 0 3 0 1 0 7
 7 1 7 1 6 7 6 6 4 4 4 0 6 2 7 5 7 7 4 7 7 0 7 6 5 3 1 5 7 1 4 4 4 4 1 5 4
 4 4 6 0 4 4 0 0 4 4 0 0 4 2 6 6 6 0 1 5 7 6 1 7 6 7 2 0 4 1 4 5 3 0 0 6 4
 7 1 2 5 7 6 0 4 0 1 4 0 0 0 5]
WCSS cost for optimal k: 3800.3406663064675


## Elbow Plot for seed 33

![Elbow Plot](./figures/elbow_kmeans_5.png)

Optimal number of clusters (k_kmeans1): 5
Cluster labels: [1 1 3 3 3 4 3 1 4 3 4 4 3 2 2 0 3 1 1 3 1 3 3 1 3 3 3 4 4 3 3 2 0 2 4 4 4
 2 1 3 4 3 4 1 0 3 1 1 1 1 3 1 0 1 0 0 3 1 3 4 2 0 3 4 3 4 0 4 4 0 4 4 4 1
 2 0 0 2 0 1 1 0 0 3 0 3 4 0 3 3 1 4 1 3 3 3 1 2 3 1 2 3 0 0 4 0 2 0 3 0 4
 4 2 4 0 3 4 0 3 3 0 0 0 3 3 4 0 4 4 1 1 4 3 1 3 0 1 2 2 4 2 3 1 1 2 2 2 0
 3 1 3 0 3 4 0 0 1 1 0 0 1 3 3 3 3 0 2 0 4 0 0 4 3 4 3 0 3 2 1 2 1 0 3 3 1
 4 0 3 2 4 3 0 1 0 2 1 0 0 0 2]
WCSS cost for optimal k: 3966.973318266988


## Elbow Plot for seed 35

![Elbow Plot](./figures/elbow_kmeans_4.png)

Optimal number of clusters (k_kmeans1): 4
Cluster labels: [2 1 0 0 0 1 2 2 1 0 2 1 0 3 3 2 0 1 2 2 2 0 0 2 0 2 0 1 1 2 0 3 3 3 1 1 1
 3 1 0 1 2 1 1 2 2 2 2 2 2 0 2 2 1 2 3 2 1 0 1 3 3 0 1 2 1 3 1 1 2 1 1 1 1
 3 3 3 2 3 2 2 3 3 0 3 2 1 2 2 0 2 1 1 0 2 0 1 3 0 2 3 0 2 2 1 2 3 3 0 3 1
 1 2 1 3 0 1 3 0 2 2 2 2 0 0 1 3 1 1 1 1 1 0 1 0 3 2 2 3 1 3 2 2 2 2 3 3 2
 2 2 2 2 2 1 2 3 1 2 3 2 1 0 0 3 0 2 2 3 1 3 2 1 0 1 2 2 2 2 1 3 2 2 2 0 2
 1 2 2 3 1 0 2 2 2 3 2 2 2 2 3]
WCSS cost for optimal k: 4057.293281791949

**We settle at $K_{kmeans1}$ = 5 due to the uniform nature of the WCSS graph and the single indisputable elbow point.**

For this we get WCSS cost as 3966.973318266988

Cluster 0:
table, mug, gym, passport, roof, stairs, bed, microwave, notebook, van, sweater, microphone, jacket, bench, bucket, feet, laptop, door, calendar, chair, ladder, candle, igloo, clock, oven, calculator, pillow, envelope, dustbin, ambulance, television, throne, tent, camera, car, loudspeaker, lantern, telephone, stove, wheel, toaster, shoe, keyboard, radio, truck, suitcase
Cluster 1:
drive, sing, dive, exit, brick, smile, bullet, bend, fly, face, climb, kneel, scream, kiss, selfie, catch, sleep, baseball, hollow, basket, empty, slide, drink, angry, lazy, hang, skate, tattoo, earth, tank, key, swim, zip, cook, basketball, arrow, walk, sunny
Cluster 2:
needle, eraser, brush, feather, spoon, pencil, knit, cigarette, flute, scissor, badminton, finger, hammer, toothbrush, screwdriver, fingerprints, teaspoon, length, sword, knife, toothpaste, comb, fork, paintbrush
Cluster 3:
deer, panda, ape, rose, helicopter, cat, carrot, fishing, bear, spider, shark, grass, giraffe, forest, lizard, frog, puppet, lake, monkey, rifle, cow, starfish, plant, sun, puppy, boat, pear, peacock, fish, saturn, fruit, grape, mouse, ant, goldfish, bird, spiderman, bee, tree, beetle, snake, rain, airplane, pizza, tomato, dragonfly, parachute, butterfly, elephant, pant, rainy, bicycle, windmill, potato, crocodile
Cluster 4:
listen, flame, sit, knock, bury, download, eat, postcard, hard, fight, call, hit, paint, far, dig, cry, run, clap, pull, clean, sad, draw, pray, arrest, email, buy, burn, fire, close, scary, book, enter, happy, loud, love, recycle, cut

---

# Gaussian Mixture Models

I initially tried performing GMM clustering using my own class using k = $K_{kmeans1}$ = 5, but ran into several challenges.

1. **Covariance Matrix Singularity**: 
   In the initial implementation, I encountered an error due to the covariance matrix being singular. A singular covariance matrix occurs when some features are linearly dependent or when there are too few data points compared to the number of features (512-dimensional data with 200 points in this case). This makes the covariance matrix non-invertible, and GMM fails because the matrix must be non-singular (and positive definite) to perform multivariate normal distribution calculations.

   To address this, I initially tried using `allow_singular=True` in `scipy.stats.multivariate_normal`. This allows the algorithm to proceed even if the covariance matrix is singular. [As it may lead to instabilities](https://stackoverflow.com/questions/35273908/scipy-stats-multivariate-normal-raising-linalgerror-singular-matrix-even-thou), I switched to **regularizing the covariance matrix**. This means adding a small value (`1e-6` in this case, can be changed using `reg_covar`) to the diagonal of the covariance matrix to ensure it's singular and positive definite. This method works better by making the matrix invertible with minimal changes to the data.

2. **Overflow Issues**:
  I encountered an overflow error during computation due to the high-dimensional data. I switched to using logarithms in the calculation. Specifically, I used the `logsumexp` function to safely compute the log of sums of exponentials, which prevents overflow issues during the expectation step (E-step). The function `_log_multivariate_normal_density` now computes the log of the Gaussian distribution for each point, and the likelihood is calculated using the `logsumexp` function.

### **Comparison with `sklearn` GMM**

After fixing the issues in my custom GMM class, I compared it with the `sklearn` GMM implementation. It seems sklearn handles singular matrices by default through preprocessing such as regularization. Both my class and the `sklearn` class successfully perform GMM clustering without errors. The log-likelihood values and the parameters (means, weights, covariances) are very similar, confirming that the core algorithm works as expected.

**To note:**
The final clustering effectively has, what can be called, hard assignments where each data point is assigned to only one cluster.The dataset  512 dimensions with only 200 data points causing posterior probabilities (membership values) to become very close to either 0 or 1. As covariance matrices are nearly degenerate (i.e., close to singular), the model may not accurately represent the true distribution of the data.The log-likelihood values for both my implementation and Sklearn are positive.

## AIC and BIC:

### My Implementation
![Large](./figures/aic_bic_gmm_large.png)
![Upto 20](./figures/aic_bic_gmm_upto_20.png)

## Sklearn
![Large](./figures/aic_bic_gmm_large_sklearn.png)
![Upto 20](./figures/aic_bic_gmm_upto_20_sklearn.png)

We note that both my code and Sklearn throw error for when we try to use more than k = 200 for GMM.

Optimal number of clusters based on BIC: 1
Optimal number of clusters based on AIC: 1

This again can be attributed to the nature of the dataset, which has more dimensions than datapoints. 

Thus, although both my and Sklearn's GMM runs on the data, it is far from ideal. PCA needs to be applied to be able to draw meaningful conclusions from the data.

**We settle at $K_{gmm1}$ = 1**

This gives us Log Likelihood: 375255.27444188285. The membership is obviously a one hot vector [1] for each class. 

```
Weights: [1]
Means: [[-1.29900086e-02, -5.21759896e-02, -1.90298252e-02,
         7.76085762e-02, -5.49704913e-02, -3.46861647e-02,
        -1.20059666e-01, -1.08713815e+00, -1.75144684e-01,
         2.18883874e-01,  1.43400622e-02, -1.11440015e-01,
        ...,
        -6.93333779e-02, -6.63823007e-02, -2.48983755e-01,
        -1.11258245e-01,  3.13115194e-02]], 
Covariances: [[[ 2.98234736e-02,  7.82239280e-05,  3.55016624e-03, ...,
         -6.82535628e-03,  2.67777455e-03, -1.17062036e-03],
        [ 7.82239280e-05,  4.33966894e-02, -2.43116468e-04, ...,
         -8.78883327e-03, -4.15580173e-04,  3.88595736e-03],
        ...,
        [ 2.67777455e-03, -4.15580173e-04,  1.66209159e-03, ...,
         -1.49421301e-02,  5.28725709e-02,  8.54919329e-04],
        [-1.17062036e-03,  3.88595736e-03,  3.46288488e-03, ...,
         -9.74189033e-03,  8.54919329e-04,  4.17133432e-02]]]
```

---

# Dimensionality Reduction and Visualization

![Large](./figures/pca_2d.png)
![Upto 20](./figures/pca_3d.png)

## Analysis of PCA Axes

**Words sorted along PCA Axis 1:**  
['peacock' 'lantern' 'crocodile' 'dragonfly' 'panda' 'giraffe' 'starfish'
 'elephant' 'sweater' 'pencil' 'windmill' 'beetle' 'toothbrush' 'eraser'
 'stairs' 'goldfish' 'bee' 'ant' 'badminton' 'pear' 'bench' 'cow' 'sun'
 'chair' 'deer' 'saturn' 'jacket' 'helicopter' 'parachute' 'flute'
 'microphone' 'tent' 'igloo' 'suitcase' 'tomato' 'feather' 'van' 'fruit'
 'gym' 'knit' 'paintbrush' 'butterfly' 'cat' 'bed' 'rifle' 'microwave'
 'tree' 'toothpaste' 'screwdriver' 'carrot' 'bicycle' 'spoon' 'grape'
 'lizard' 'ladder' 'pillow' 'throne' 'needle' 'ambulance' 'teaspoon'
 'fork' 'comb' 'length' 'feet' 'frog' 'stove' 'roof' 'candle' 'scissor'
 'mug' 'spider' 'grass' 'car' 'toaster' 'fishing' 'laptop' 'shark'
 'potato' 'pant' 'boat' 'airplane' 'spiderman' 'bucket' 'ape' 'forest'
 'table' 'mouse' 'television' 'door' 'loudspeaker' 'oven' 'camera'
 'finger' 'rainy' 'lake' 'monkey' 'telephone' 'brush' 'calculator'
 'calendar' 'plant' 'puppet' 'envelope' 'wheel' 'sword' 'keyboard' 'bear'
 'cigarette' 'skate' 'rain' 'rose' 'clock' 'pizza' 'dustbin'
 'fingerprints' 'snake' 'passport' 'bird' 'climb' 'shoe' 'basketball'
 'arrow' 'hammer' 'truck' 'baseball' 'walk' 'kneel' 'fish' 'knife' 'radio'
 'basket' 'scream' 'brick' 'swim' 'zip' 'puppy' 'tattoo' 'drive' 'kiss'
 'notebook' 'bend' 'sunny' 'sit' 'bullet' 'pull' 'dig' 'fly' 'exit' 'tank'
 'face' 'run' 'hollow' 'dive' 'smile' 'earth' 'empty' 'key' 'clap' 'eat'
 'selfie' 'postcard' 'paint' 'catch' 'drink' 'sing' 'slide' 'cook' 'angry'
 'arrest' 'recycle' 'bury' 'knock' 'book' 'sleep' 'far' 'email' 'lazy'
 'draw' 'enter' 'cut' 'hang' 'loud' 'call' 'sad' 'clean' 'close' 'flame'
 'fight' 'fire' 'pray' 'burn' 'happy' 'buy' 'hard' 'scary' 'download'
 'listen' 'love' 'cry' 'hit']

**Words sorted along PCA Axis 2:**  
['sad' 'cat' 'run' 'cow' 'eat' 'bee' 'panda' 'ape' 'dig' 'potato' 'sun'
 'pant' 'van' 'car' 'gym' 'tomato' 'buy' 'sit' 'camera' 'ant' 'rifle'
 'bed' 'elephant' 'clap' 'calendar' 'pencil' 'feet' 'mug' 'fly'
 'crocodile' 'bear' 'fruit' 'cry' 'starfish' 'lantern' 'pull' 'television'
 'cut' 'peacock' 'puppy' 'deer' 'frog' 'arrest' 'selfie' 'monkey' 'smile'
 'hard' 'face' 'bird' 'zip' 'plant' 'grape' 'happy' 'calculator' 'earth'
 'ambulance' 'angry' 'laptop' 'shoe' 'scary' 'rose' 'tree' 'fish'
 'helicopter' 'chair' 'radio' 'fire' 'roof' 'sweater' 'boat' 'close' 'far'
 'kneel' 'sleep' 'enter' 'hit' 'lake' 'lazy' 'parachute' 'clean' 'forest'
 'pray' 'rain' 'truck' 'jacket' 'spider' 'tank' 'burn' 'hang' 'love'
 'sunny' 'bench' 'table' 'snake' 'shark' 'pizza' 'email' 'grass' 'tattoo'
 'lizard' 'book' 'kiss' 'tent' 'microphone' 'rainy' 'flame' 'bicycle'
 'butterfly' 'listen' 'mouse' 'empty' 'paint' 'stove' 'drink' 'hammer'
 'pear' 'door' 'fight' 'swim' 'postcard' 'puppet' 'walk' 'draw' 'finger'
 'giraffe' 'toaster' 'download' 'exit' 'cigarette' 'spiderman' 'bend'
 'beetle' 'oven' 'recycle' 'climb' 'wheel' 'slide' 'baseball' 'goldfish'
 'cook' 'loud' 'bury' 'dive' 'pillow' 'throne' 'length' 'fork' 'fishing'
 'ladder' 'eraser' 'basketball' 'keyboard' 'suitcase' 'carrot' 'candle'
 'hollow' 'call' 'saturn' 'sing' 'stairs' 'brick' 'clock' 'catch' 'bucket'
 'key' 'drive' 'scissor' 'windmill' 'dragonfly' 'skate' 'knock' 'bullet'
 'envelope' 'knit' 'scream' 'igloo' 'airplane' 'microwave' 'toothpaste'
 'basket' 'arrow' 'knife' 'fingerprints' 'passport' 'telephone' 'sword'
 'dustbin' 'needle' 'notebook' 'flute' 'feather' 'comb' 'brush' 'spoon'
 'toothbrush' 'loudspeaker' 'screwdriver' 'teaspoon' 'badminton'
 'paintbrush']

**Words sorted along PCA Axis 3:**  
['laptop' 'chair' 'calendar' 'calculator' 'television' 'mug' 'bench'
 'envelope' 'pencil' 'microphone' 'stove' 'microwave' 'keyboard' 'pillow'
 'camera' 'table' 'suitcase' 'radio' 'bed' 'toaster' 'loudspeaker'
 'throne' 'van' 'tent' 'gym' 'email' 'oven' 'car' 'telephone' 'toothbrush'
 'sweater' 'notebook' 'potato' 'call' 'stairs' 'jacket' 'comb' 'book'
 'postcard' 'sit' 'eraser' 'passport' 'ambulance' 'fork' 'brush' 'pull'
 'dig' 'door' 'ladder' 'eat' 'lantern' 'toothpaste' 'sad' 'dustbin'
 'clock' 'candle' 'rifle' 'run' 'shoe' 'teaspoon' 'selfie' 'feet' 'bucket'
 'buy' 'flute' 'cut' 'cigarette' 'recycle' 'draw' 'length' 'listen'
 'drink' 'roof' 'truck' 'sun' 'paintbrush' 'knit' 'download' 'pant' 'clap'
 'wheel' 'knock' 'tattoo' 'igloo' 'badminton' 'cook' 'finger' 'exit'
 'hammer' 'brick' 'screwdriver' 'arrest' 'kneel' 'enter' 'smile' 'tomato'
 'spoon' 'key' 'slide' 'basketball' 'face' 'knife' 'bend' 'sing'
 'fingerprints' 'bury' 'burn' 'paint' 'happy' 'fruit' 'boat' 'baseball'
 'far' 'basket' 'scissor' 'zip' 'walk' 'close' 'tank' 'parachute' 'hit'
 'cry' 'drive' 'empty' 'bicycle' 'loud' 'kiss' 'scary' 'skate' 'sword'
 'sleep' 'catch' 'needle' 'helicopter' 'cat' 'plant' 'tree' 'pizza'
 'fight' 'cow' 'lazy' 'puppet' 'fire' 'windmill' 'love' 'sunny' 'hard'
 'grass' 'hang' 'hollow' 'pear' 'peacock' 'bullet' 'scream' 'clean'
 'carrot' 'pray' 'angry' 'fishing' 'mouse' 'rose' 'feather' 'flame'
 'earth' 'ape' 'swim' 'rain' 'grape' 'panda' 'bee' 'rainy' 'ant' 'climb'
 'dive' 'saturn' 'lake' 'arrow' 'starfish' 'airplane' 'forest' 'fly'
 'elephant' 'goldfish' 'crocodile' 'puppy' 'beetle' 'fish' 'snake' 'bird'
 'bear' 'spiderman' 'deer' 'monkey' 'butterfly' 'shark' 'spider' 'giraffe'
 'lizard' 'frog' 'dragonfly']

### PCA Axis 1: Abstraction Level
This axis appears to represent a continuum from concrete, tangible objects to abstract concepts and actions.

- Starts with: Concrete nouns (e.g., peacock, lantern, crocodile)
- Ends with: Abstract verbs and concepts (e.g., love, cry, hit)


### PCA Axis 2: Animacy and Mobility

This axis seems to capture the distinction between things that can move or have agency and those that are typically stationary or used as tools.

- Starts with: Living creatures and dynamic objects (e.g., sad, cat, run, cow)
- Ends with: Static tools and small objects (e.g., paintbrush, teaspoon, screwdriver)


### PCA Axis 3: Human-Centric vs. Nature-Centric
This axis seems to move from human-made, indoor items from natural, outdoor elements.

- Starts with: Man-made, indoor objects (e.g., laptop, chair, calendar)
- Ends with: Natural, outdoor elements (e.g., butterfly, shark, spider, giraffe)

By observation I was able to identify 4 distinct clusters. Thus, $$k_2 = 4$$

# PCA + KMeans Clustering

K-Means on $k_2 = 4$  
Clusters:
3 1 3 3 3 1 3 3 1 3 3 1 3 2 2 0 3 3 3 3 3 3 3 3 3 3 3 1 1 3 3 2 0 2 1 1 1
 2 3 3 1 3 1 3 0 3 3 3 3 3 3 3 0 1 0 0 3 1 3 1 2 0 3 1 3 1 0 3 1 0 3 1 3 1
 2 0 0 3 0 3 3 0 0 3 0 3 1 0 3 3 3 1 3 3 3 3 1 2 3 1 2 3 0 0 1 0 2 0 3 0 1
 1 2 1 0 3 1 0 3 3 0 0 0 3 3 1 0 1 1 1 1 1 3 1 3 0 3 2 2 1 2 3 3 3 3 2 2 0
 3 3 3 0 3 1 0 0 2 3 0 0 3 3 3 3 3 0 2 0 1 0 0 1 3 1 3 0 3 2 1 2 3 0 3 3 3
 1 0 3 2 1 3 3 3 0 2 3 0 0 0 2
WCSS cost for k=4: 4063.222058671721

## Scree Plot for Optimal Dimensions:

![Cumulative](./figures/scree_plot_cumulative.png)
![Individual](./figures/scree_plot_individual.png)

**Optimal number of dimensions based on 90% explained variance: 107**

## Elbow plot with reduced dataset

![Elbow](figures/elbow_kmeans_optimal_clusters_5.png)

Observing the elbow plot,$k_{kmeans3} = 5$

Performing K-Means clustering with k=5 on the reduced dataset, we get:

Cluster labels for reduced dataset (k=5): [1 1 3 3 3 4 3 1 4 3 4 4 3 2 2 0 3 1 0 3 1 3 3 1 3 3 3 4 4 3 3 2 0 2 4 4 4
 2 1 3 4 3 4 1 0 3 1 1 1 1 3 1 0 1 0 2 3 1 3 4 2 0 3 1 3 4 0 4 4 0 4 4 1 1
 2 0 0 2 0 1 1 0 0 3 0 3 4 0 3 3 1 4 1 3 3 3 1 2 3 1 2 3 0 2 4 0 2 0 3 2 4
 4 2 4 0 3 4 0 3 3 0 0 0 3 3 4 0 4 4 1 1 4 3 1 3 0 1 2 2 4 2 3 1 1 2 2 2 0
 3 1 3 0 3 4 0 0 1 1 0 0 1 3 3 3 3 0 2 0 4 0 0 4 3 4 3 0 3 2 1 2 1 0 3 3 1
 4 0 3 2 4 3 0 1 0 2 1 0 0 0 2]
WCSS cost for reduced dataset (k=5): 3530.4813079576106

---

# PCA + GMM

GMM on $k_2 = 4$ gives log likelihood 544089.8358658948

## AIC/BIC with reduced dataset

### Using My Implementation

Optimal number of clusters based on BIC: 4  
Optimal number of clusters based on AIC: 6

![AIC-BIC-My_Version](figures/aic_bic_pca_gmm_4_6.png)

### Using Sklearn

Optimal number of clusters based on BIC: 4  
Optimal number of clusters based on AIC: 6

![AIC-BIC-SKLEARN](figures/aic_bic_pca_gmm_sklearn_4_6.png)

**We take $k_{gmm3} = argmin_k BIC = 4$**   

Applying $k_{gmm3} = 4$  
**Log Likelihood using my implementation**: 57082.45751057781
**Log Likelihood using Sklearn**: 57393.64792527589

Reconstruction Error: 0.03835961161605422
Explained Variance Ratio: 0.1325
Reconstruction Error: 0.03692622888597155
Explained Variance Ratio: 0.1649

--- 

# K- Means Cluster Analysis

Number of clusters = k_kmeans1 = 5
Cluster 0
table, mug, gym, passport, roof, stairs, bed, microwave, notebook, van, sweater, microphone, jacket, bench, bucket, feet, laptop, door, calendar, chair, ladder, candle, igloo, clock, oven, calculator, pillow, envelope, dustbin, ambulance, television, throne, tent, camera, car, loudspeaker, lantern, telephone, stove, wheel, toaster, shoe, keyboard, radio, truck, suitcase
Cluster 1
drive, sing, dive, exit, brick, smile, bullet, bend, fly, face, climb, kneel, scream, kiss, selfie, catch, sleep, baseball, hollow, basket, empty, slide, drink, angry, lazy, hang, skate, tattoo, earth, tank, key, swim, zip, cook, basketball, arrow, walk, sunny
Cluster 2
needle, eraser, brush, feather, spoon, pencil, knit, cigarette, flute, scissor, badminton, finger, hammer, toothbrush, screwdriver, fingerprints, teaspoon, length, sword, knife, toothpaste, comb, fork, paintbrush
Cluster 3
deer, panda, ape, rose, helicopter, cat, carrot, fishing, bear, spider, shark, grass, giraffe, forest, lizard, frog, puppet, lake, monkey, rifle, cow, starfish, plant, sun, puppy, boat, pear, peacock, fish, saturn, fruit, grape, mouse, ant, goldfish, bird, spiderman, bee, tree, beetle, snake, rain, airplane, pizza, tomato, dragonfly, parachute, butterfly, elephant, pant, rainy, bicycle, windmill, potato, crocodile
Cluster 4
listen, flame, sit, knock, bury, download, eat, postcard, hard, fight, call, hit, paint, far, dig, cry, run, clap, pull, clean, sad, draw, pray, arrest, email, buy, burn, fire, close, scary, book, enter, happy, loud, love, recycle, cut
Cluster labels: [1 1 3 3 3 4 3 1 4 3 4 4 3 2 2 0 3 1 1 3 1 3 3 1 3 3 3 4 4 3 3 2 0 2 4 4 4
 2 1 3 4 3 4 1 0 3 1 1 1 1 3 1 0 1 0 0 3 1 3 4 2 0 3 4 3 4 0 4 4 0 4 4 4 1
 2 0 0 2 0 1 1 0 0 3 0 3 4 0 3 3 1 4 1 3 3 3 1 2 3 1 2 3 0 0 4 0 2 0 3 0 4
 4 2 4 0 3 4 0 3 3 0 0 0 3 3 4 0 4 4 1 1 4 3 1 3 0 1 2 2 4 2 3 1 1 2 2 2 0
 3 1 3 0 3 4 0 0 1 1 0 0 1 3 3 3 3 0 2 0 4 0 0 4 3 4 3 0 3 2 1 2 1 0 3 3 1
 4 0 3 2 4 3 0 1 0 2 1 0 0 0 2]
WCSS cost for optimal k: 3966.973318266988
_______________________
Number of clusters = K_2 = 4
Cluster 0
table, mug, gym, passport, roof, stairs, bed, microwave, notebook, van, sweater, microphone, jacket, bench, bucket, feet, laptop, door, calendar, chair, ladder, candle, igloo, clock, oven, calculator, pillow, envelope, dustbin, ambulance, television, throne, tent, camera, car, loudspeaker, lantern, telephone, stove, wheel, toaster, keyboard, radio, truck, suitcase
Cluster 1
sing, listen, flame, knock, bury, download, eat, postcard, hard, fight, call, selfie, catch, hit, paint, far, cry, clap, sleep, clean, sad, slide, drink, draw, pray, arrest, email, buy, burn, fire, close, angry, lazy, scary, hang, book, enter, happy, loud, love, cook, recycle, cut
Cluster 2
needle, eraser, brush, feather, spoon, pencil, knit, flute, scissor, badminton, finger, hammer, toothbrush, screwdriver, teaspoon, length, key, sword, knife, toothpaste, comb, fork, paintbrush
Cluster 3
drive, deer, panda, ape, rose, dive, helicopter, sit, cat, carrot, exit, brick, fishing, smile, bear, spider, bullet, shark, grass, giraffe, forest, lizard, bend, frog, puppet, fly, lake, face, climb, kneel, scream, monkey, kiss, rifle, cow, starfish, plant, dig, run, pull, cigarette, baseball, hollow, sun, puppy, boat, pear, basket, empty, peacock, fish, saturn, fruit, grape, mouse, ant, goldfish, bird, spiderman, bee, tree, beetle, skate, snake, tattoo, earth, fingerprints, rain, tank, airplane, pizza, swim, zip, tomato, dragonfly, parachute, butterfly, elephant, pant, rainy, basketball, bicycle, windmill, arrow, potato, crocodile, shoe, walk, sunny
Cluster labels: [3 1 3 3 3 1 3 3 1 3 3 1 3 2 2 0 3 3 3 3 3 3 3 3 3 3 3 1 1 3 3 2 0 2 1 1 1
 2 3 3 1 3 1 3 0 3 3 3 3 3 3 3 0 1 0 0 3 1 3 1 2 0 3 1 3 1 0 3 1 0 3 1 3 1
 2 0 0 3 0 3 3 0 0 3 0 3 1 0 3 3 3 1 3 3 3 3 1 2 3 1 2 3 0 0 1 0 2 0 3 0 1
 1 2 1 0 3 1 0 3 3 0 0 0 3 3 1 0 1 1 1 1 1 3 1 3 0 3 2 2 1 2 3 3 3 3 2 2 0
 3 3 3 0 3 1 0 0 2 3 0 0 3 3 3 3 3 0 2 0 1 0 0 1 3 1 3 0 3 2 1 2 3 0 3 3 3
 1 0 3 2 1 3 3 3 0 2 3 0 0 0 2]
WCSS cost for optimal k: 4063.222058671721

---

Number of clusters = k_kmeans3 = 5
Cluster 0
table, mug, gym, passport, roof, stairs, bed, microwave, notebook, van, sweater, microphone, jacket, bench, bucket, feet, laptop, door, calendar, chair, ladder, candle, igloo, clock, oven, calculator, pillow, envelope, dustbin, ambulance, television, throne, tent, camera, car, loudspeaker, lantern, telephone, stove, wheel, toaster, shoe, keyboard, radio, truck, suitcase  

Cluster 1
drive, sing, dive, exit, brick, smile, bullet, bend, fly, face, climb, kneel, scream, kiss, selfie, catch, sleep, baseball, hollow, basket, empty, slide, drink, angry, lazy, hang, skate, tattoo, earth, tank, key, swim, zip, cook, basketball, arrow, walk, sunny  

Cluster 2
needle, eraser, brush, feather, spoon, pencil, knit, cigarette, flute, scissor, badminton, finger, hammer, toothbrush, screwdriver, fingerprints, teaspoon, length, sword, knife, toothpaste, comb, fork, paintbrush  

Cluster 3
deer, panda, ape, rose, helicopter, cat, carrot, fishing, bear, spider, shark, grass, giraffe, forest, lizard, frog, puppet, lake, monkey, rifle, cow, starfish, plant, sun, puppy, boat, pear, peacock, fish, saturn, fruit, grape, mouse, ant, goldfish, bird, spiderman, bee, tree, beetle, snake, rain, airplane, pizza, tomato, dragonfly, parachute, butterfly, elephant, pant, rainy, bicycle, windmill, potato, crocodile
Cluster 4
listen, flame, sit, knock, bury, download, eat, postcard, hard, fight, call, hit, paint, far, dig, cry, run, clap, pull, clean, sad, draw, pray, arrest, email, buy, burn, fire, close, scary, book, enter, happy, loud, love, recycle, cut
Cluster labels: [1 1 3 3 3 4 3 1 4 3 4 4 3 2 2 0 3 1 1 3 1 3 3 1 3 3 3 4 4 3 3 2 0 2 4 4 4
 2 1 3 4 3 4 1 0 3 1 1 1 1 3 1 0 1 0 0 3 1 3 4 2 0 3 4 3 4 0 4 4 0 4 4 4 1
 2 0 0 2 0 1 1 0 0 3 0 3 4 0 3 3 1 4 1 3 3 3 1 2 3 1 2 3 0 0 4 0 2 0 3 0 4
 4 2 4 0 3 4 0 3 3 0 0 0 3 3 4 0 4 4 1 1 4 3 1 3 0 1 2 2 4 2 3 1 1 2 2 2 0
 3 1 3 0 3 4 0 0 1 1 0 0 1 3 3 3 3 0 2 0 4 0 0 4 3 4 3 0 3 2 1 2 1 0 3 3 1
 4 0 3 2 4 3 0 1 0 2 1 0 0 0 2]
WCSS cost for optimal k: 3966.973318266988

## Analysis of k=5 Clustering (= $K_{kmeans1}$ = $K_{kmeans3}$)

1. Cluster 0: Indoor/Household Objects
   - Examples: table, mug, microwave, bed, chair, oven
   - Theme: Primarily indoor and household items

2. Cluster 1: Actions 
   - Examples: drive, sing, dive, smile, climb, sleep

3. Cluster 2: Small Tools and Instruments
   - Examples: needle, eraser, brush, spoon, pencil, hammer

4. Cluster 3: Animals and Nature
   - Examples: deer, panda, rose, cat, grass, tree

5. Cluster 4: Activities
   - Examples: listen, flame, bury, fight, pray, love

## Analysis of k=4 Clustering (= $K_2$)

1. Cluster 0: Indoor/Household Objects
   - Examples: table, mug, microwave, bed, chair, oven

2. Cluster 1: Actions
   - Examples: sing, listen, flame, fight, pray, love

3. Cluster 2: Small Tools and Instruments
   - Examples: needle, eraser, brush, spoon, pencil, hammer
   - Theme: Identical to Cluster 2 in k=5

4. Cluster 3: Concrete Objects and Actions (most of which are also nouns)
   - Examples: drive, deer, panda, rose, smile, boat

## Evaluation

1. Interpretability:
   - k=5 offers more distinct and interpretable clusters, separating actions, abstract concepts, and concrete objects more clearly.
   - k=4 combines some categories, making interpretation slightly less straightforward.

2. Homogeneity:
   - k=5 appears to have more homogeneous clusters, especially in separating abstract actions from concrete actions.
   - k=4 has more mixed clusters, particularly in Cluster 3.

3. WCSS (Within-Cluster Sum of Squares) Cost:
   - k=5: 3966.97
   - k=4: 4063.22
   - The lower WCSS for k=5 indicates tighter, more compact clusters.

4. Granularity:
   - k=5 provides a finer granularity, which might be beneficial for more detailed analysis.
   - k=4 offers a simpler, more general categorization.

# GMM Cluster Analysis

---

# Hierarchical Clustering

---

# KNN + PCA

![Cumulative](./figures/spotify_scree_plot_cumulative.png)
![Individual](./figures/spotify_scree_plot_individual.png)

Optimal number of dimensions based on 85% explained    variance: 9  
Reconstruction Error: 0.13712198288339003  
Explained Variance Ratio: 0.8629  
Classification Task Scores  
Accuracy: 0.2561  
  Precision (macro): 0.2340  
  Recall (macro): 0.2472  
  F1-Score (macro): 0.2405  
  Precision (micro): 0.2561  
  Recall (micro): 0.2561  
  F1-Score (micro): 0.2561  
**Time Taken:** 65.20476651191711  
  
Classification Task Scores  
Accuracy: 0.2207  
  Precision (macro): 0.1941  
  Recall (macro): 0.2135  
  F1-Score (macro): 0.2033  
  Precision (micro): 0.2207  
  Recall (micro): 0.2207  
  F1-Score (micro): 0.2207  
**Time Taken:** 53.93823790550232