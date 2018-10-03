# entity-embedding-experiment

Deep learning has proven to be outperforming the traditional machine learning methods in the unstructured data domains like image, audio and text while the improved versions of traditional methods like Gradient Boosted Trees and Random Forests are still dominating in the structured(tabular) data domains. The use of such traditional methods require a lot of manual feature engineering usually coupled with specific domain expertise. Interstingly, there's a newfound love for deep learning in tabular data, specifically in the presence of categorical variables. The idea of using a deep network to a data set with categorical variables has been brought to the attention of the deep learning community by the [Entity Embeddings of Categorical Variables](https://arxiv.org/abs/1604.06737) paper, which describes a winning solution to a [kaggle competition](https://www.kaggle.com/c/rossmann-store-sales).

This project details an experiment done for the categorical entity embedding task using the famous [Adult Data Set](https://archive.ics.uci.edu/ml/datasets/adult), where the goal is to predit if the income of a particular person will be less than or greater than 50K USD. Once a model is built an API is developed to use the model assuming a case where the user sends a file and expects the predictions.

## Method of execution

Walk through the notebooks in the below order:

* [1. Data Preprocessing.ipynb](https://github.com/akilat90/entity-embedding-experiment/blob/master/1.%20Data%20Preprocessing.ipynb): Basic Data Preprocessing.
* [2. Entity Embedding Network.ipynb](https://github.com/akilat90/entity-embedding-experiment/blob/master/2.%20Entity%20Embedding%20Network.ipynb): Training the entity embedding network, saving models for future use.
* [3. Visualizing Learned Embeddings.ipynb](https://github.com/akilat90/entity-embedding-experiment/blob/master/3.%20Visualizing%20Learned%20Embeddings.ipynb): Visualizing the learned categorical embeddings.
* [4. API Usage Demo.ipynb](https://github.com/akilat90/entity-embedding-experiment/blob/master/4.%20API%20Usage%20Demo.ipynb): Demonstrating the usage of the created API.

Most of the code for the network and the related processing is in [embed_helpers.py](embed_helpers.py).

Model serving code is available at [server.py](https://github.com/akilat90/entity-embedding-experiment/blob/master/server.py)

---

### Local run instructions

You can use the [Floydhub docker image](https://github.com/floydhub/dl-docker). You may need to follow these steps:

    docker pull floydhub/dl-docker:cpu
    docker run -it -p 8888:8888 -p 5000:5000 -v /sharedfolder:/root/sharedfolder floydhub/dl-docker:cpu bash
    
Then, update the libraries in the requirements.txt using `pip install -r requirements.txt`. Also [make sure that keras has the tensorflow back end](https://keras.io/backend/).

If a local run without docker is desired, make sure all the dependencies in requirements.txt are installed with keras backend dependency mentioned above.

---    

* Since the models and the data are already available in the repo, you don't need to run the notebooks 1-3 if you just want to test the API; If so, just run the 4th notebook 

* Clone the repo and walk through the notebooks in the order of their numbering for a complete walkthrough.

### Note:

I'd say this is a baseline experiment and there's so much room to improve upon this setup. Some examples are:

* Trying different model architectures/parameters.
* using better cross-validation strategies. 
* Using the learned categorical embeddings in a different model.
* The problem is an imbalnced class problem where the class 0 : class 1 ratio is about 1:3. Evaluation metrics other than accuracy (like AUC ROC/PRC, F1 score needs to be considered.

### Some interesting related posts:

1. [Applying deep learning to Related Pins](https://medium.com/the-graph/applying-deep-learning-to-related-pins-a6fee3c92f5e) by Pinterest.
2. [Instacart's categorical embedding blog post](https://tech.instacart.com/deep-learning-with-emojis-not-math-660ba1ad6cdc)
3. [fast.ai blog on deep learning for tabular data](http://www.fast.ai/2018/04/29/categorical-embeddings/)

This repo is mostly inspired by the [repo](https://github.com/entron/entity-embedding-rossmann) by the authors of the original paper and [this kernel](https://www.kaggle.com/aquatic/entity-embedding-neural-net/code) in kaggle.
