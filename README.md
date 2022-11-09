# IMDB-Topic-Modeling-and-Sentiment-Classification
## LDA Topic Modeling and Deep Learning-based Sentiment Classification on IMDB Movie Reviews

### Introduction:

In this project, I presented an experimental approach to discover abstract topics using text mining and perform sentiment classification using a deep learning-based framework on the IMDB Movie Review dataset. I used the **LDA** (Latent Dirichlet Allocation) topic modeling to classify movie reviews to a particular topic. During the experiment, the LDA model extracted 10 topics from the IMDB data and allocate the most relevant topic to each review based on their overall subject. In the next phase, I used the pre-trained **BERT** (Bidirectional Encoder Representations from Transformers) model to perform binary classification on IMDB reviews based on their sentiment polarities. I split the entire dataset into 80:20 ratio for training and validation purposes. I used batch processing to reduce the computational complexity of the model during the training phase. Finally, the model achieved **91.83%** training accuracy and **89.94%** validation accuracy during sentiment classification.

### Data Description:

Here the description of the database has been presented.

- **review** - IMDB Movie reviews.

- **sentiment** - Sentiment polarity of the reviews (e.g., Positive and Negative).

### Workflow of the Project:

    i. Preprocessing of IMDB Reviews,

        ii. Popular Token Identification,
    
            iii. n-gram Analysis,
            
                 iv. LDA Topic Modeling,
                 
                      v. Sentiment Classification,
                 
                          vi. Sentiment Prediction,
                          
                              vii. Sentiment Prediction on user-end reviews.
              
### Required Packages
Please install the following packages to execute all the codes.

- **pandas**==1.3.4
- **numpy**==1.20.3
- **tweet-preprocessor**==0.5.0
- **seaborn**==0.11.2
- **matplotlib**==3.4.3
- **networkx**==2.6.3
- **wordcloud**==1.8.1
- **nltk**==3.6.5
- **scikit-learn**==1.1.1
- **tqdm**==4.62.3
- **keras**==2.7.0
- **tensorflow**==2.7.0
- **transformers**==4.18.0              
                    
#### - By Arunava Kumar Chakraborty
