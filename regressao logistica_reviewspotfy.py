#%%
#%%
import nltk
import pandas as pd
from nltk import tokenize
nltk.download('punkt')
nltk.download('stopwords')
import seaborn as sns
from sklearn import metrics
from string import punctuation
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english')
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Leitura da base de dados
data = pd.read_csv('review_v4.csv')

#%%
# Quantidade de sentimentos positivos e negativos na base de dados
data["Rating"] = data["Rating"].replace([1, 2],[0, 0])
data["Rating"] = data["Rating"].replace([4, 5],[1, 1])

#%%

data_clean = data.dropna(inplace=False, axis=0)
data=data_clean
#%%
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

    #return [w for w in words if w not in stop_words]
data['tratamento_3'] = data.tratamento_2.apply(rem_stopwords)

#Stemização - Reduz a palavra a sua raiz morfologica
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

#%%
def wordCloud_pos(text, column_text,bg_color):
    text_positive = text.query("Rating == 1")
    all_words = ' '.join([text for text in text_positive[column_text]])
    word_Cloud = WordCloud(background_color=bg_color, width = 800, height = 500,
                                max_font_size=110, collocations=False).generate(all_words)
    plt.figure(figsize=(10,7))
    plt.imshow(word_Cloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    
wordCloud_pos(data, 'tratamento_4', 'white')
#%%
def wordCloud_neg(text, column_text, bg_color):
    texto_negative = text.query("Rating == 0")
    all_words = ' '.join([texto for texto in texto_negative[column_text]])
    word_Cloud = WordCloud(background_color=bg_color, width = 800, height = 500,
                                max_font_size=110, collocations=False).generate(all_words)
    plt.figure(figsize=(10,7))
    plt.imshow(word_Cloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
#%%
wordCloud_neg(data, 'tratamento_4', 'black')

# BOW
#%% Classificar texto
def classificar_texto(texto, coluna_texto, coluna_classificacao):
    vetorizar = CountVectorizer(lowercase=False, max_features=50)
    bag_of_words = vetorizar.fit_transform(texto[coluna_texto])
    treino, teste, classe_treino, classe_teste = train_test_split(bag_of_words,
                                                                 texto[coluna_classificacao], test_size=0.3,
                                                                 random_state = 42)
    
    regressao_logistica = LogisticRegression(solver="lbfgs")
    regressao_logistica.fit(treino, classe_treino)
    
    return regressao_logistica.score(teste, classe_teste)
result = classificar_texto(data, 'tratamento_4', 'Rating')

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
#%% TF-IDF
regressao_logistica = LogisticRegression(solver="lbfgs")
tfidf = TfidfVectorizer(lowercase=False, max_features=50)
tfidf_bruto = tfidf.fit_transform(data["tratamento_4"])
treino, teste, classe_treino, classe_teste = train_test_split(tfidf_bruto,
                                                              data["Rating"], test_size=0.3,
                                                              random_state = 42)
regressao_logistica.fit(treino, classe_treino)
acuracia_tfidf_bruto = regressao_logistica.score(teste, classe_teste)
print(acuracia_tfidf_bruto)

#%% 
tfidf = TfidfVectorizer(lowercase=False)
vetor_tfidf = tfidf.fit_transform(data["tratamento_4"])
treino, teste, classe_treino, classe_teste = train_test_split(vetor_tfidf,
                                                              data["Rating"], test_size=0.3,
                                                              random_state = 42)
regressao_logistica.fit(treino, classe_treino)
acuracia_tfidf = regressao_logistica.score(teste, classe_teste)
print(acuracia_tfidf)
# # %%
clf = regressao_logistica.fit(treino, classe_treino)
predicted = clf.predict(teste)

#%%
print("acuracia ", metrics.accuracy_score(classe_teste, predicted))
print("matriz confusao ", metrics.confusion_matrix(classe_teste, predicted))
print("classificacao report ", metrics.classification_report(classe_teste, predicted))
print("precision score ", metrics.precision_score(classe_teste, predicted))
print("recall score ", metrics.recall_score(classe_teste, predicted))
print("recall score ", metrics.precision_recall_fscore_support(classe_teste, predicted))
#%%

y_pred = clf.predict(teste)
cm = confusion_matrix(classe_teste, y_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()
#%%
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
y_probabilities = clf.predict_proba(teste)[:, 1]  # Probabilidades da classe positiva
fpr, tpr, _ = roc_curve(classe_teste, y_probabilities)
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
#%%
pesos = pd.DataFrame(
    regressao_logistica.coef_[0].T,
    index = tfidf.get_feature_names_out()
)

pesos.nlargest(10,0)
pesos.nsmallest(10,0)

#%%

