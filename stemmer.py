# stem means base and root of a word
from nltk.stem.porter import *
from nltk.stem.lancaster import *
from nltk.stem.wordnet import WordNetLemmatizer
import hazm as hz

#get the stem of every token
sample_tokens = ["connection" , "connecting" , "connect" , "connected" , "disconnect"]
main_tokes = ["lovely" , "decentrelized","better" ,"information" ,"language" ,"goes" , "better", "disable" , "did"]


porter = PorterStemmer()
stemlist = []
for token in sample_tokens:
    s = porter.stem(token)
    stemlist.append(s)


#another stemmer: overstemm
lanclist = []
lancaster = LancasterStemmer()
for token in sample_tokens:
    s = lancaster.stem(token)
    lanclist.append(s)


#combination both

for token in main_tokes:
    p = porter.stem(token)
    l = lancaster.stem(token)

#lemmatization :  meaningful root of a word
#using part of speech or role of a word in a text
#for all token we can define part of speech : A for adjective, v for verb and...

lemmatizer = WordNetLemmatizer()
lemlist = []
for token in main_tokes:
    l = lemmatizer.lemmatize(token)
    p = porter.stem(token)
    lemlist.append(l)

#persian
main_token_fa = [" کتابی","کنابم"]

stemmer_fa = hz.Stemmer()
lemmatizer_fa = hz.Lemmatizer()
normalizer = hz.Normalizer()

#at first you should normalize the text
for token in main_tokes:
    token = normalizer.Normalize(token)
    S = stemmer_fa.stem(token)
    l = lemmatizer_fa.lemmatize(token)



