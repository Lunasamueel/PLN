#%%
import pandas as pd
import numpy as np
import re
import seaborn as sns
import nltk
from nltk import tokenize
nltk.download('stopwords')
nltk.download('punkt')
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from string import punctuation
stemmer = SnowballStemmer('english')
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
# %%

# Leitura da base de dados
data = pd.read_csv('review_v4.csv')

#%%
# Quantidade de sentimentos positivos e negativos na base de dados
data["Rating"] = data["Rating"].replace([1, 2],[0, 0])
data["Rating"] = data["Rating"].replace([4, 5],[1, 1])
# Quantidade de sentimentos positivos e negativos na base de dados
data.Rating.value_counts()

data_clean = data.dropna(inplace=False, axis=0)
data=data_clean

data.head()


# RETIRAR A PONTUAÇÃO
#%%
def remove_punctuation(review):
    
    textpunctuation = list()
    for item in punctuation:
        textpunctuation.append(item)

    processed_sentence = ''
    
    for word in review:
        if word not in textpunctuation:
            processed_sentence = processed_sentence + word
        else:
            processed_sentence = processed_sentence + ' '

    return processed_sentence
data['tratamento_1'] = data.Review.apply(remove_punctuation)

# TRANFORMAR A RESENHA EM MINUSCULO
#%%
def to_lower(review):
    return review.lower()
data['tratamento_2'] = data.tratamento_1.apply(to_lower)

#%%

# Criar função apra o grafico de pareto
def pareto(text, column_text, quantity):
    all_words = ' '.join([text for text in text[column_text]])
    space_token = tokenize.WhitespaceTokenizer()
    frequency = nltk.FreqDist(space_token.tokenize(all_words))
    df_frequency = pd.DataFrame({"Word": list(frequency.keys()),
                                 "Frequency": list(frequency.values())})
    df_frequency = df_frequency.nlargest(columns = "Frequency", n = quantity)
    plt.figure(figsize=(12,8))
    ax = sns.barplot(data = df_frequency, x= "Word", y = "Frequency", color = 'black')
    ax.set(ylabel = "Contagem")
    plt.show()   

pareto(data, "tratamento_2", 8)

# REMOVER STOPWORDS
#%%
def rem_stopwords(text):
    stop_words = nltk.corpus.stopwords.words("english")
    words = word_tokenize(text)
    review = ""
    for w in words:
        if w not in stop_words:
            review += w + " "
    return review

data['tratamento_3'] = data.tratamento_2.apply(rem_stopwords)

#LEMATIZAÇÃO - Reduz a palavra a sua raiz morfologica
# %%
def stemming(text):
    stop_words = nltk.corpus.stopwords.words("english")
    processed_sentence = list()
    
    for review in text.split():
        punctuation_token = tokenize.WordPunctTokenizer()
        palavras_texto = punctuation_token.tokenize(review)
        new_sentence = list()
        for word in palavras_texto:
            if word not in stop_words:
                new_sentence.append(stemmer.stem(word))
        processed_sentence.append(' '.join(new_sentence))
    new_sentence = ""
    for word in processed_sentence:
        new_sentence += word + " "
    return new_sentence.rstrip()
data['tratamento_4'] = data.tratamento_3.apply(stemming)

pareto(data, "tratamento_4", 8)

x = data["tratamento_4"]
y = data['Rating']

# BOW
vetorizar = CountVectorizer(lowercase = False, max_features = 50)
bag_of_words = vetorizar.fit_transform(x.values.astype('str'))
trainx, testx, trainy, testy = train_test_split(bag_of_words, y, random_state = 9)

def classificar_texto(texto, coluna_texto, coluna_classificacao):
    vetorizar = CountVectorizer(lowercase=False, max_features=50)
    bag_of_words = vetorizar.fit_transform(texto[coluna_texto].values.astype('str'))
    trainx, testx, trainy, testy = train_test_split(bag_of_words, texto[coluna_classificacao], random_state = 9)

    mnb = MultinomialNB()
    mnb.fit(trainx, trainy)
    return mnb.score(testx, testy)
acuracia = classificar_texto(data, "tratamento_4", "Rating")
print("accuracy ",acuracia)
acuracia = classificar_texto(data, "Review", "Rating")
acuracia_tratamento1 = classificar_texto(data, "tratamento_1", "Rating")
acuracia_tratamento2 = classificar_texto(data, "tratamento_2", "Rating")
acuracia_tratamento3 = classificar_texto(data, "tratamento_3", "Rating")
acuracia_tratamento4 = classificar_texto(data, "tratamento_4", "Rating")
print(f"acuracia {acuracia}")
print(f"acuracia_tratamento1 {acuracia_tratamento1}")
print(f"acuracia_tratamento2 {acuracia_tratamento2}")
print(f"acuracia_tratamento3 {acuracia_tratamento3}")
print(f"acuracia_tratamento4 {acuracia_tratamento4}")
#%%

tfidf = TfidfVectorizer()
tfidf_bruto = tfidf.fit_transform(x)

print('the tf idf features')
df = pd.DataFrame(tfidf_bruto.toarray(), columns=tfidf.get_feature_names_out())
#%%
xtrain, xtest, ytrain, ytest = train_test_split(tfidf_bruto, y, test_size=0.3, random_state=2)
#%%

clf = MultinomialNB().fit(xtrain, ytrain)
predicted = clf.predict(xtest)
#%%
from sklearn import metrics
print("acuracia ", metrics.accuracy_score(ytest, predicted))
print("matriz confusao ", metrics.confusion_matrix(ytest, predicted))
print("classificacao report ", metrics.classification_report(ytest, predicted))
print("precision score ", metrics.precision_score(ytest, predicted, average="micro"))
print("recall score ", metrics.recall_score(ytest, predicted, average="micro"))
print("recall score ", metrics.precision_recall_fscore_support(ytest, predicted))
# %%

y_pred = clf.predict(xtest)
cm = confusion_matrix(ytest, y_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()
#%%

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
y_probabilities = clf.predict_proba(xtest)[:, 1]  # Probabilidades da classe positiva
fpr, tpr, _ = roc_curve(ytest, y_probabilities)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (área = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()



# %%
