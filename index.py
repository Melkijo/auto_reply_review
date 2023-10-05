import streamlit as st
import pandas as pd
import math
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import csv
import numpy as np

st.title('Auto Reply Chatbot')
uploaded_file = st.file_uploader("Choose a file")
dataframe = None;

def ganti_isi_column(data_lama,data_baru):
  for i in range(len(data_lama)) :
    data_lama[i] = data_baru[i]
  print(data_lama)

def preprocessing(document_to_be_scored):
  # Tokenize Dataset
  pisah = {}
  for i in range(len(document_to_be_scored)):
    pisah[i] = document_to_be_scored[i].split() #melakukan split

  # Menghilangkan stopword memakai library nltk
  stopword=['dengan'] # custom list stopword

  removed = {}
  hasil = []
  list_stopword = stopwords.words('indonesian') # list stopword dari NLTK

  # Tambahkan stopword
  for i in stopword:
    list_stopword.append(i)
#   print(list_stopword,"\n")

  # listStopword = set(stopwords.words('indonesian')) #list stopword
  hitung = 0 #untuk menghitung frekuensi
  for i in pisah.values():
    for j in i:
      if j not in list_stopword: #jika setiap kata tidak ada di list stopword
        hasil.append(j)
    removed[hitung] = hasil[:]
    del hasil[:]
    hitung=hitung+1

  # Memakai library sastarawi
  # Memakai library sastrawi untuk melakukan stemming
  factory = StemmerFactory() #membuat objek stemmer
  stemmer = factory.create_stemmer()
  hitung = 0

  hasil=[]
  hasil2 = {}

  for i in removed.values():
    for j in i:
      # Hasil Stemming
      hasil.append(stemmer.stem(j))
    hasil2[hitung] = hasil[:]
    del hasil[:]
    hitung = hitung+1
  hasil =""
  korpus2 =[]
  for i in hasil2.values():
    hasil =""
    for j in i:
      hasil = hasil+ " " +j
    korpus2.append(hasil)
  #Timpa hasil
  ganti_isi_column(document_to_be_scored,korpus2)


def compute_tfidf(documents):
    # Step 1: Tokenization
    def tokenize(document):
        return document.lower().split()

    # Step 2: Create a vocabulary
    vocabulary = set()
    for doc in documents:
        vocabulary.update(tokenize(doc))

    vocabulary = sorted(vocabulary)

    # Step 3: Compute TF (Term Frequency)
    # print("\nIterasi pada setiap data:")
    # count=1
    tf_matrix = []
    for doc in documents:
        tokens = tokenize(doc)
        tf = [tokens.count(word) for word in vocabulary]
        # print("Data Ke-",count,":",tokens)
        tf_matrix.append(tf)
        # count+=1


    # Step 4: Compute IDF (Inverse Document Frequency)
    idf = []
    N = len(documents)
    for word in vocabulary:
        doc_count = sum([1 for doc in documents if word in tokenize(doc)])
        idf.append(math.log(N / (1 + doc_count)))


    # Step 5: Compute TF-IDF
    tfidf_matrix = []
    for tf_vector in tf_matrix:
        tfidf = [tf * idf_value for tf, idf_value in zip(tf_vector, idf)]
        tfidf_matrix.append(tfidf)
    
    return vocabulary, tf_matrix, idf, tfidf_matrix

def tfidf (review_handphone):
    for i in range(len(review_handphone)):
        review_handphone['label'][i] = review_handphone['label'][i].lower()
        review_handphone['review'][i] = review_handphone['review'][i].lower()

    documents = review_handphone['review']

    vocabulary,tf_matrix,idf,tfidf_matrix = compute_tfidf(documents)

    csv_file_path = 'tfidf_features.csv'
    # Save the TF-IDF matrix to a CSV file
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row (vocabulary)
        writer.writerow(vocabulary)

        # Write each TF-IDF vector as a row
        for tfidf_vector in tfidf_matrix:
            writer.writerow(tfidf_vector)
    st.success('TF-IDF Features berhasil dibuat!', icon="✅")
    return review_handphone

from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.metrics.pairwise import cosine_similarity

def compute_cosine_similarity(text1, text2):

    # list
    list_text = [text1, text2]

    # converts text into vectors with the TF-IDF
    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform(list_text)
    tfidf_text1, tfidf_text2 = vectorizer.transform([list_text[0]]), vectorizer.transform([list_text[1]])

    # computes the cosine similarity
    cs_score = cosine_similarity(tfidf_text1, tfidf_text2)

    return np.round(cs_score[0][0],5)

def consine_similarity(query,review_handphone, limit):
    # Preprocessing
    factory = StemmerFactory() #membuat objek stemmer
    stemmer = factory.create_stemmer()
    query =stemmer.stem(query)
    query = query.lower()
    similarity = {}
    i =0

    for doc in review_handphone['review']: #similarities antara input dan review
        similarity[i]= compute_cosine_similarity(query,doc)
        i = i+1
    output = sorted(similarity.items(), key=lambda x:x[1], reverse=True)

    i =0
    positive =0
    negative =0

    # make dataframe to store data index, similarity, label, review
    data_output = pd.DataFrame(columns=['index','similarity','label','review'])

    for k, v in output:
    #jika nilai scoring dibawah 0.01 skip
        if v < 0.01 :continue 
        if(i >= limit): break
        try:
            similarity = "{:.5f}".format(v)
            data_output = data_output.append({'index':k,'similarity':similarity,'label':review_handphone['label'][k],'review':review_handphone['review'][k]},ignore_index=True)
            if(review_handphone['label'][k] == 'positif'):
                positive+=1
            else:
                negative+=1    

        except:
            continue
        i=i+1
    

    if(positive>=negative):
        st.write("Balasan: Terima kasih broh #respect")
        st.balloons()
    else:
        st.write("Balasan: Mohon maaf atas ketidaknyamanannya, kami akan berusaha lebih baik lagi :)")

    st.write( "Nilai positif prediksi  = ",positive)
    st.write("Nilai negative prediksi = ",negative)
    st.dataframe(data_output)
    # precision(positive,negative,label_document_1,15)

review_handphone = None
if uploaded_file is not None:
            # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    review_handphone = tfidf(dataframe)

    query = st.text_input('Masukkan review handphone')
    limit = st.slider('Jumlah limit yang ingin ditampilkan', 1, len(review_handphone), 5)
    button = st.button('Submit')
    if(button):
        consine_similarity(query,review_handphone, limit)
    

   



