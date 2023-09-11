#bag of word is a matrix of 0,1 which determine a word is in a doc or not (#doc * #word]
#Tf-IDF : giving value of every word in each doc concidering the Frequency of word in each doc and whole docs
#Tf : Term Frequency in doc / total number of the term in doc
#IDF : Inverse Doc Frequency : log ( number of doc / number of doc contain the term)
# TF-IDF = TF*IDF
import re
from nltk.corpus import stopwords
from nltk imort sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
text = ""
stimmer = PorterStemmer()

#cleaning text
sentences = sent_tokenize ( text)

sentlist=[]
for sent in sentences:
    #substitude every thing except letter with space
    review = re.sub("[^a-zA-Z]" , " " , sent)
    # removing a single letter appeared by deleting other symbols
    review = re.sub("\b[a-zA-z]\b" , " ",review)
    review = review.lower() #lower case letter
    review = review.split()
    # remove stop words like : of , and , the , a
    review = [stimmer.stem(word) for word in review if not word in set(stopwords.word('english'))]
    #making sentence from word list again
    sentence = " ".join(review)
    sentlist.append(sentence)


#vectorize by scikit learn
#vectorize in text means feature extraction
cv = CountVectorizer()
#output is 0 and 1 in array
Bag_Word = cv.fit_transform(sentlist).toarray() # output is a sparse matrix

#tfidf vectorizer
tf = TfidfVectorizer()
#output is a sparse matrix with float number
tfidf = tf.transform(sentlist).toarray()

#convert sparse matrix to a dense matrix




