# entity-embedding-experiment

Deep learning has proven to be outperforming the traditional machine learning methods in the unstructured data domains like image, audio and text while the improved versions of traditional methods like Gradient Boosted Trees and Random Forests are still dominating in the structured(tabular) data domains. The use of such traditional methods require a lot of manual feature engineering usually coupled with specific domain expertise. Interstingly, there's a newfound love for deep learning in tabular data, specifically in the presence of categorical variables. The idea of using a deep network to a data set with categorical variables has been brought to the attention of the deep learning community by the [Entity Embeddings of Categorical Variables](https://arxiv.org/abs/1604.06737) paper, which describes a winning solution to a [kaggle competition](https://www.kaggle.com/c/rossmann-store-sales).

This project details an experiment done for the categorical entity embedding task using the famous [Adult Data Set](https://archive.ics.uci.edu/ml/datasets/adult), where the goal is to predit if the income of a particular person will be less than or greater than 50K USD. Once a model is built an API is developed to use the model assuming a case where the user sends a file and expects the predictions.

---

## A note about the network architecture

As defined in the `make_embedding_network()` function in [embed_helpers.py](https://github.com/akilat90/entity-embedding-experiment/blob/master/embed_helpers.py), the network accepts a list of input arrays, where each categorical column is represented by a one array and all the numeric columns in another one array. Considering a single instance of the network input for the Adult data set, the input should look like below:

    workclass_cat 	education_cat 	marital.status_cat 	occupation_cat 	relationship_cat 	race_cat 	sex_cat 	native.country_cat 	age 	fnlwgt 	education.num 	capital.gain 	capital.loss 	hours.per.week
       [2] 	        [11] 	            [6] 	        [3]     	         [1]     	     [4]     	   [0]     	           [38]         [82 	132870 	        9 	            0 	        4356 	    18]

Note how the categorical variables (columns that have `_cat` in their name) have a single input (a number) and all other numeric variables are collected together to a one array (`[age 	fnlwgt 	education.num 	capital.gain 	capital.loss 	hours.per.week]`) - these numeric values will ideally be normalized when feeding to the network.

The integer each category is assigned is the Label encoded value for the category value. The below is an illustration when the `workclass_cat` value is `2`. Note that the `workclass_cat` column has seven different categories: 

    ['Federal-gov', 'Local-gov', 'Private', 'Self-emp-inc',
       'Self-emp-not-inc', 'State-gov', 'Without-pay']

![wc](https://github.com/akilat90/entity-embedding-experiment/blob/master/img/work_class_input.png)

The W matrix (of size 7 x 3) is the embedding matrix for the categorical variable `workclass_cat`. Since there are 7 different categories for the `workclass_cat`, there are seven rows in the embedding matrix - one row for each category. 3 is the dimensionality of the vector space that we wish to map the category's values, which is a parameter of our choice.

The output from the three units are then passed to the second layer. This process happens the same way to all the categorical variables whereas the numeric columns just get fed to the second layer directly.

Once the network is trained for a classification task, we can obtain the W matrices for each category, that happen to be the weights of the network and categorical embeddings at the same time.

---

## Local setup instructions

0. Download/clone this project to a local directory, say `<your-local-directory>`.
1. Install [Docker](https://docs.docker.com/install/#upgrade-path) if you don't have it already.
2. You can use the [Floydhub docker image](https://github.com/floydhub/dl-docker). Follow these steps:

        docker pull floydhub/dl-docker:cpu
        docker run -it -p 8888:8888 -p 5000:5000 -v <your-local-directory>:/root/sharedfolder floydhub/dl-docker:cpu bash
    
    This will mount `<your-local-directory>` to `/root/sharedfolder` in the container.
    
 3. Then, update the libraries in the [requirements.txt](https://github.com/akilat90/entity-embedding-experiment/blob/master/requirements.txt) using `pip install -r requirements.txt` (this may take some time). Also [make sure that keras has the tensorflow back end](https://keras.io/backend/).
 4. Launch a jupyter notebook inside the container at `/root/sharedfolder` by running `jupyter notebook --ip=*`
 5. You can now access the project and it's notebooks from `localhost:8888` from your host's browser. 

If a local run without docker is desired, make sure all the dependencies in requirements.txt are installed with keras backend dependency mentioned above.

---    

## Method of execution

Walk through the notebooks in the below order:

* [1. Data Preprocessing.ipynb](https://github.com/akilat90/entity-embedding-experiment/blob/master/1.%20Data%20Preprocessing.ipynb): Basic Data Preprocessing.
* [2. Entity Embedding Network.ipynb](https://github.com/akilat90/entity-embedding-experiment/blob/master/2.%20Entity%20Embedding%20Network.ipynb): Training the entity embedding network, saving models for future use.
* [3. Visualizing Learned Embeddings.ipynb](https://github.com/akilat90/entity-embedding-experiment/blob/master/3.%20Visualizing%20Learned%20Embeddings.ipynb): Visualizing the learned categorical embeddings.
* [4. API Usage Demo.ipynb](https://github.com/akilat90/entity-embedding-experiment/blob/master/4.%20API%20Usage%20Demo.ipynb): Demonstrating the usage of the created API.

Most of the code for the network and the related processing is in [embed_helpers.py](embed_helpers.py).

Model serving code is available at [server.py](https://github.com/akilat90/entity-embedding-experiment/blob/master/server.py)

* Since the models and the data are already available in the repo, you don't need to run the notebooks 1-3 if you just want to test the API; If so, just run the 4th notebook 

* Walk through the notebooks in the order of their numbering for a complete walkthrough.

---

### Note:

I'd say this is a baseline experiment and there's so much room to improve upon this setup. Some examples are:

* Trying different model architectures/parameters.
* using better cross-validation strategies. 
* Using the learned categorical embeddings in a different model.
* The problem is an imbalnced class problem where the class 0 : class 1 ratio is about 1:3. Evaluation metrics other than accuracy (like AUC ROC/PRC, F1 score needs to be considered.
* Accomodating an unknown category for each column to avoid the embedding matrix look-up errors.

### Some related posts:

1. [Applying deep learning to Related Pins](https://medium.com/the-graph/applying-deep-learning-to-related-pins-a6fee3c92f5e) by Pinterest.
2. [Instacart's categorical embedding blog post](https://tech.instacart.com/deep-learning-with-emojis-not-math-660ba1ad6cdc)
3. [fast.ai blog on deep learning for tabular data](http://www.fast.ai/2018/04/29/categorical-embeddings/)

This repo is mostly inspired by the [repo](https://github.com/entron/entity-embedding-rossmann) by the authors of the original paper and [this kernel](https://www.kaggle.com/aquatic/entity-embedding-neural-net/code) in kaggle.
