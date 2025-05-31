import pandas as pd
import numpy as np
import nltk

df = pd.read_csv("C:/Users/Telka LLC/Desktop/Jupytr/spam_app/spam.csv", encoding="windows-1252")
df.head(5)

df.shape

df.info()

#data cleaning
#drop the last 3 columns
df.drop(columns =['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace= True)
df.sample(5)

#renaming the columns
df.rename(columns={'v1':'target','v2':'text'},inplace=True)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

df['target'] = encoder.fit_transform(df['target'])
df.head(5)

df.isnull().sum()

#checking for duplicates
df.duplicated().sum()

#removing the dulicates
df = df.drop_duplicates(keep='first')
df.shape

df['target'].value_counts()

#EDA
import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(),labels = ['ham','spam'])
plt.show()


nltk.download('punkt',download_dir= "C:/Users/Telka LLC/AppData/Roaming/nltk_data")
nltk.data.path.append("C:/Users/Telka LLC/AppData/Roaming/nltk_data")
df['num_characters']= df['text'].apply(len) #gives length in terms of characters used
df.head()

#num of words: breaking sms based on work
df['text'].apply(lambda x:nltk.word_tokenize(x))

df['num_of_words']=df['text'].apply(lambda x:len(nltk.word_tokenize(x)))
print(df.head(5))

#number of setences
df['num_of_sentences']= df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))
df.head(5)

df[['num_characters','num_of_words','num_of_sentences']].describe()
#roughly 2 sentences per messages, 78.97 characters and 18 words

#this is for ham messages
#just 70 chracters per message
df[df['target']==0][['num_characters','num_of_words','num_of_sentences']].describe()

#this is for spam messages
#137 characters on an average (These messages are bigger than ham messages)
df[df['target']==1][['num_characters','num_of_words','num_of_sentences']].describe()

import seaborn as sns
sns.histplot(df[df['target']==0]['num_characters'])
#plt.show()

sns.histplot(df[df['target']==1]['num_characters'],color = 'red')
plt.show()
#spam messages has more characters on an avg as compared to spam

#histogram on words
sns.histplot(df[df['target']==0]['num_of_words'])
#plt.show()

sns.histplot(df[df['target']==1]['num_of_words'],color = 'red')
plt.show()

#To see the relationship between all the variables
sns.pairplot(df,hue='target')
plt.show()


#getting correlation coefficient
numeric_df = df.select_dtypes(include='number')
sns.heatmap(numeric_df.corr(), annot=True)
plt.show()
#high correlation of num sentences with num character
#we need only 1 since there is an issue of multi colinearity 
#we will keep num characters due to its high correlation with target

'''Data Preprocessing: lowercase, tokenize, remove special characters and punctuation, stemming'''
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords.words('english')
import string
string.punctuation

#lowercase
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum(): #if it is alphanumeric and removing special characters
            y.append(i)
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
              y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

#transform_text("I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.") 
#df['text'][10]
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
#ps.stem("loving")

#making the transformed column basd on the function
df['transformed_text']= df['text'].apply(transform_text)
df.head(5)
df.columns


from wordcloud import WordCloud
wc = WordCloud(width =500, height=500,min_font_size=10,background_color='white')
spam_wc= wc.generate(df[df['target']==1]['transformed_text'].str.cat(sep=" "))#spam
plt.imshow(spam_wc)
plt.axis("off")
plt.show()


wc = WordCloud(width =500, height=500,min_font_size=10,background_color='white')
ham_wc= wc.generate(df[df['target']==0]['transformed_text'].str.cat(sep=" "))#spam
plt.imshow(ham_wc)
plt.axis("off")
plt.show()

#if I want to know how many exact words are top 30 to see more frequent
spam_corpus =[]
for msg in df[df['target']==1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)
len(spam_corpus)

#getting top 30 words from spam
from collections import Counter
pd.DataFrame(Counter(spam_corpus).most_common(30))

#getting the top 30 words from ham
ham_corpus =[]
for msg in df[df['target']==0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)
len(ham_corpus)


spam_words = pd.DataFrame(Counter(spam_corpus).most_common(30), columns=['word', 'count'])

# Now pass explicitly as keyword arguments
plt.figure(figsize=(10, 5))
sns.barplot(x='word', y='count', data=spam_words)
plt.xticks(rotation=90)
plt.title("Top 30 Most Common Spam Words")
plt.tight_layout()
plt.show()


ham_words = pd.DataFrame(Counter(ham_corpus).most_common(30), columns=['word', 'count'])

# Now passing explicitly as keyword arguments
plt.figure(figsize=(10, 5))
sns.barplot(x='word', y='count', data=ham_words)
plt.xticks(rotation=90)
plt.title("Top 30 Most Common Spam Words")
plt.tight_layout()
plt.show()

#model building (doing Naives Bayes because it works well with this kind of data)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x = cv.fit_transform(df['transformed_text']).toarray()

y = df['target'].values
y

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=2)


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score

gb = GaussianNB()
bnb = BernoulliNB()
mb = MultinomialNB()

gb.fit(x_train,y_train)
y_pred1 = gb.predict(x_test)
print(accuracy_score(y_test,y_pred1)) #.88
print(confusion_matrix(y_test, y_pred1))
print(precision_score(y_test,y_pred1))#really poor: 53%

#for multinomial NB
mb.fit(x_train,y_train)
y_pred_mb = mb.predict(x_test)
print(accuracy_score(y_test,y_pred_mb))  #excellent 96%
print(confusion_matrix(y_test, y_pred_mb))
print(precision_score(y_test,y_pred_mb))#not that good score of 83%

bnb.fit(x_train,y_train) #this seems to be better
y_pred_bnb = bnb.predict(x_test)
print(accuracy_score(y_test,y_pred_bnb))#excellent accuracy of 97%
print(confusion_matrix(y_test, y_pred_bnb))
print(precision_score(y_test,y_pred_bnb)) #good precision score of 97%


#lets see if insteda of bag of words, TF-IDF gives better result
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()
x_tf = tf.fit_transform(df['transformed_text']).toarray()

y_tf = df['target'].values
y_tf

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x_tf,y_tf, test_size=0.2, random_state=2)

gb_tf = GaussianNB()
bnb_tf = BernoulliNB()
mb_tf = MultinomialNB()

gb_tf.fit(x_train,y_train)
y_pred1_tf = gb_tf.predict(x_test)
print(accuracy_score(y_test,y_pred1_tf)) #87%
print(confusion_matrix(y_test, y_pred1_tf))
print(precision_score(y_test,y_pred1_tf)) #really poor with 52%

#for multinomial NB
mb_tf.fit(x_train,y_train)
y_pred_mb_tf = mb_tf.predict(x_test) 
print(accuracy_score(y_test,y_pred_mb_tf)) #95% 
print(confusion_matrix(y_test, y_pred_mb_tf))
print(precision_score(y_test,y_pred_mb_tf))# excellent score of 1 no False Positive

bnb_tf.fit(x_train,y_train)
y_pred_bnb_tf = bnb_tf.predict(x_test)
print(accuracy_score(y_test,y_pred_bnb_tf)) #97%
print(confusion_matrix(y_test, y_pred_bnb_tf))
print(precision_score(y_test,y_pred_bnb_tf)) #97%

'''CHOOSING TF-IDF WITH MB: We can either go with Bernoulli NB since both have amazing precision and accuracy scores but since
precision matters most, going with Multinomial NB with 1 presion score'''

#trying out other models:
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

dt = DecisionTreeClassifier(max_depth = 5)
rf = RandomForestClassifier(n_estimators = 54, random_state=2)
gbc = GradientBoostingClassifier(random_state = 2)
xgb = XGBClassifier(max_depth = 5, learning_rate = 0.1)

accuracy_num = []
precision_num = []
model_selection = {
#    'DT' : dt,
    'RF' : rf,
    'GBC': gbc,
    'XGB': xgb,
    'MN' : mb_tf
    
    }

def train_test(model_selection,x_train,x_test,y_train,y_test):
    model_selection.fit(x_train,y_train)
    y_pred = model_selection.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test,y_pred)
    
    return accuracy,precision

'''test = train_test(rf,x_train,x_test,y_train,y_test)
print(test)

print("X_train:", x_train.shape)
print("y_train:", y_train.shape)
print("X_test:", x_test.shape)
print("y_test:", y_test.shape)
y.dtype
x.dtype
print(x.shape)
print(y.shape)'''

for name, model_selection in model_selection.items():
    model_selection.fit(x_train, y_train)
    y_pred = model_selection.predict(x_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    
    print(f"{name} → Accuracy: {acc:.4f}, Precision: {prec:.4f}")
    accuracy_num.append(acc)
    precision_num.append(prec)
    
#These are the Accuracies and Precisions:
'''
    RF → Accuracy: 0.9739, Precision: 1.0000 <-- this is probabby the best model
    GBC → Accuracy: 0.9574, Precision: 0.9123
    XGB → Accuracy: 0.9671, Precision: 0.9561
    MN → Accuracy: 0.9594, Precision: 1.0000
'''