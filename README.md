## Multi-granular Legal Topic Classification on Greek Legislation

Code and data repo for the paper: [Multi-granular Legal Topic Classification on Greek Legislation](https://arxiv.org/abs/2109.15298)\
presented at NLLP 2021 workshop co-located with EMNLP 2021.

Dataset is available at HuggingFace 🤗: https://huggingface.co/datasets/greek_legal_code

***
### Abstract

In this work, we study the task of classifying legal texts written in the Greek language. We introduce and make publicly available a novel dataset based on Greek
legislation, consisting of more than 47 thousand official, categorized Greek legislation resources. We experiment with this dataset and evaluate a battery of
advanced methods and classifiers, ranging from traditional machine learning and RNN-based methods to state-of-the-art Transformer-based methods.

We show that recurrent architectures with domain-specific word embeddings offer improved overall performance while being competitive even to transformer-based 
models. Finally, we show that cutting-edge multilingual and monolingual transformer-based models brawl on the top of the classifiers’ ranking, making us question
the necessity of training monolingual transfer learning models as a rule of thumb.

To the best of our knowledge, this is the first time the task of Greek legal text classification is considered in an open research project, while also Greek is a
language with very limited NLP resources in general.

***
### Code

Contact me via email for code access.

Word2Vec embeddings available at: http://legislation.di.uoa.gr/publications/ner_word2vec

**Note:**  
NLP training scripts based on "Large-Scale Multi-Label Text Classification on EU Legislation" project. For full code and project structure, follow lmtc-eurlex57k project instructions at: https://github.com/iliaschalkidis/lmtc-eurlex57k
