# NLP (Natural Language Processing) Toolkit
Base knowledge for NLP 

## Table of Contents
1. [Vectorization](#vectorization)
2. [Embedding](#embedding)
3. [Stemming](#stemming)
4. [TF-IDF (Term Frequency-Inverse Document Frequency)](#tf-idf)
5. [Tokenization](#tokenization)
6. [Bag of Words](#bag-of-words)

## Vectorization
Vectorization is the process of converting text data into numerical vectors that can be used in machine learning algorithms. In NLP, this often involves techniques like one-hot encoding or word embeddings to represent words or phrases as numerical values.

## Embedding
Word embedding is a technique that represents words as dense vectors in a continuous vector space. Word embeddings capture semantic relationships between words, making them useful for various NLP tasks like sentiment analysis, text classification, and machine translation.

## Stemming
Stemming is the process of reducing words to their root or base form. It helps in reducing variations of words to a common form, which can improve the performance of text analysis algorithms.

## TF-IDF (Term Frequency-Inverse Document Frequency)
TF-IDF is a numerical statistic used to evaluate the importance of a word within a document relative to a collection of documents (corpus). It's often used for information retrieval and text mining tasks to identify the most relevant words in a document.

The formula for calculating TF-IDF is as follows:
**TF-IDF(t, d, D) = TF(t, d) * IDF(t, D)**
Where:
- `TF(t, d)` (Term Frequency): Measures how frequently a term `t` appears in a document `d`. It's often computed as the count of term `t` in document `d`.
- `IDF(t, D)` (Inverse Document Frequency): Measures the rarity of term `t` across the entire corpus `D` of documents. It's calculated as:
**IDF(t, D) = log(N / (1 + DF(t, D)))**
Where `N` is the total number of documents in the corpus `D`, and `DF(t, D)` is the number of documents containing the term `t`.
In practice, TF-IDF helps identify words or phrases that are unique or important within a specific document while downweighting common words that appear in many documents.

## Tokenization
Tokenization is the process of breaking text into individual words or tokens. It's a fundamental step in NLP and is crucial for tasks like text analysis, language modeling, and information retrieval.

## Bag of Words
The Bag of Words (BoW) model is a simple representation of text data that counts the frequency of words in a document without considering their order. It's a basic and effective method for text classification and sentiment analysis.
