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

Model serving code is available at server.py

---

Since the models and the data are already available in the repo, you don't need to run the notebooks 1-3 if you just want to test the API; If so, just run the 4th notebook 

Clone the repo and walk through the notebooks in the order of their numbering for a complete walkthrough. The project equires a keras installation with tensorflow backend,  pandas, numpy , matplotlib, flask and requests packages. (I'll update a requirements.txt later)




### Note:

I'd say this is a baseline experiment and there's so much room to improve upon this setup. Some examples are: Trying different model architectures/parameters, using better cross-validation strategies and using the learned categorical embeddings in a different model.

### Some interesting related posts:

1. [Applying deep learning to Related Pins](https://medium.com/the-graph/applying-deep-learning-to-related-pins-a6fee3c92f5e) by Pinterest.
2. [Instacart's categorical embedding blog post](https://tech.instacart.com/deep-learning-with-emojis-not-math-660ba1ad6cdc)
