# Perpustakaan untuk Manipulasi Data
import pandas as pd
import re
import string
import numpy as np
from tqdm import tqdm

# Perpustakaan untuk NLP Preprocessing Text
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Perpustakaan untuk Abaikan Warning
# import warnings
# from pandas.core.common import SettingWithCopyWarning
# warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
# warnings.simplefilter(action="ignore", category=RuntimeWarning)
# warnings.simplefilter(action="ignore", category=UserWarning)
# warnings.simplefilter(action="ignore", category=FutureWarning)


# Membuat Stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# ----------------Dapatkan stopword dari stopword NLTK ---------
# Dapatkan stopword bahasa Indonesia
list_stopwords = stopwords.words('indonesian')
# ------------------ Menambahkan stopword secara manual ----------
# Menambahkan kata untuk dihapus
list_stopwords.extend(['yg', 'dg', 'rt', 'dgn', 'ny', 'di', 'amp', 'duh', 'nya', 'nih', 'sih','si', 'tau', 'tdk',
                       'tuh', 'utk', 'ya', 'sdh', 'aja', 'n', 't', 'wkwkwk', 'nyg', 'hehe', 'pen', 'u', 'nan', 'loh',
                       '&amp', 'yah', 'dong', 'lah', 'eh', 'deh','sya','blm', 'mah', 'le','mu', 'la',
                       'kek', 'krn', 'udh', 'dng' , 'yaaa', 'dll', 'cemna', 'kelen', 'vi', 'ps', 'dlm', 'klo', 'tp',
                       'trus', 'gmn', 'mgkn', 'tsb', 'bp', 'bkn', 'jg', 'sbg', 'gt', 'broo', 'yyy', 'lu', 'kh', 'ngo',
                       'tyt', 'tlg', 'kpd','jd', 'sbg', 'dpt','sdg','nu', 'cm', 'spt','bro', 'gan', 'ter', 'ads', 'bs',
                       'sm', 'tu', 'vs', 'tl', 'gra', 'mmg', 'mw', 'ehm', 'blom', 'klau', 'lha', 'gg', 'ajah', 'sihhh',
                       'nieh','ef', 'ama', 'lg', 'sj', 'elu', 'da', 'jakaa', 'hrus', 'olh', 'sjak','bgt', 'et', 'bae',
                       'cih','ku','sy', 'gw','bet', 'ah', 'kyk', 'bbrp','lk', 'dri', 'ta', 'ttg', 'lho', 'hayooo', 'rb',
                       'em', 'aaee', 'kl', 'ywp', 'nich', 'dhi', 'kn', 'teu', 'ma', 'yo', 'wae', 'ora', 'son', 'klu', 'iye',
                       'emg', 'drpd', 'nih', 'minn', 'x', 'hi', 'hai', 'km', 'gtgt', 'ltlt', 'xfxdxdxf', 'xdxdxdxdxd',
                       'xdxdxdxd', 'lot', 'goog', 'xd', 'xdxdxd', 'yahh', 'xdxdxd', 'pk', 'ter', 'xf', 'bedebes', 'ckkk', 'tidak'
                       ])

# Ubah daftar menjadi kamus stopwords
list_stopwords = set(list_stopwords)

# Function untuk menjalankan tiap-tiap function yang ada dibawah
def text_preprocessing(text): 
    # Menjalankan function menghilangkan karakter spesial dan melakukan lowercase
    text = menghapus_karakter_special(text)
    # Menjalankan function menghilangkan karakter emoticon/emoji
    text = menghapus_emotikon(text)

    # Melakukan tokenizing
    tokens = word_tokenize(text)

    # Menjalankan function mengubah kata slang menjadi kata baku (normalisasi teks)
    slang_word = normalisasi_kata(tokens)
    # Menjalankan function melakukan penanganan negasi pada kata
    negation = penanganan_negasi(slang_word)

    # Melalukan stopwords -> menghilangkan kata yang bermakna kurang penting
    stopwords = [token for token in negation if token not in list_stopwords]

    # Melakukan stemming -> mengubah kata menjadi kata dasarnya
    stemmed_tokens = [stemmer.stem(token) for token in stopwords]
    return " ".join(stemmed_tokens).strip()


# Proses menghilangkan karakter spesial dan melakukan lowercase
def menghapus_karakter_special(data_tweet):
    # Menambahakan spasi setelah titik atau koma
    data_tweet =  re.sub(r"(?<=[.,])(?=[^\s])", r" ", str(data_tweet))
    # Hapus karakter non-ascii dari string
    data_tweet = re.sub(r"[^\x00-\x7f]",r" ", str(data_tweet))
    # Ganti 2+ titik dengan spasi
    data_tweet = re.sub(r"\.{2,}", " ", str(data_tweet))
    # Hapus baris baru
    data_tweet = str(data_tweet).replace("\\n", "")
    # Hapus username @
    data_tweet = re.sub(r"@\w+", "", str(data_tweet))
    # Hapus hashtags
    data_tweet = re.sub(r"#", "", str(data_tweet))
    # Hapus karakter huruf tunggal
    data_tweet = re.sub(r"\b[a-zA-Z]\b", "", str(data_tweet))
    # Hapus angka
    data_tweet = re.sub("[0-9]+", "", str(data_tweet))
    # Hapus url
    data_tweet = re.sub(r"http\S+", "", str(data_tweet))
    # Hapus spasi biasa
    data_tweet = str(data_tweet).strip(' "\'')
    # Hapus spasi/jarak yang berlebihan
    data_tweet = re.sub(r"\s+", " ", str(data_tweet))
    # Hapus tanda baca
    data_tweet = str(data_tweet).translate(str.maketrans("","",string.punctuation))
    # Hapus url yang tidak lengkap
    return str(data_tweet).replace("http://", " ").replace("https://", " ")


# Proses Menghilangkan Emoticon
def menghapus_emotikon(data_tweet):
    emoji_pattern = re.compile("["
                          u"\U0001F600-\U0001F64F"  # emoticons
                          u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                          u"\U0001F680-\U0001F6FF"  # transport & map symbols
                          u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                          u"\U00002500-\U00002BEF"  # chinese char
                          u"\U00002702-\U000027B0"
                          u"\U00002702-\U000027B0"
                          u"\U000024C2-\U0001F251"
                          u"\U0001f926-\U0001f937"
                          u"\U00010000-\U0010ffff"
                          u"\u2640-\u2642"
                          u"\u2600-\u2B55"
                          u"\u200d"
                          u"\u23cf"
                          u"\u23e9"
                          u"\u231a"
                          u"\ufe0f"  # dingbats
                          u"\u3030"
                          "]+", flags=re.UNICODE)
    data_tweet = emoji_pattern.sub(r"", str(data_tweet))
    return emoji_pattern.sub(r"", str(data_tweet))


# Membaca kamus slang dalam bentuk file csv
kamus_normalisasi_kata = pd.read_excel(r"dataset/kamus-kata.xlsx")

# Buat variabel dengan format dictionary yang kosong, dimana nantinya akan menyimpan hasil normalisasi kata
slang_word_dict = {}

for index, row in kamus_normalisasi_kata.iterrows():
    if row[0] not in slang_word_dict:
        slang_word_dict[row[0]] = row[1]

# Function untuk menerapkan normalisasi kata dari dataframe review
def normalisasi_kata(document):
    return [slang_word_dict[term] if term in slang_word_dict else term for term in document]


# Proses untuk menangani masalah negasi pada kata
def penanganan_negasi(data_tweet):
    negasi_data_tweet = []
    for i in range(len(data_tweet)):
        word = data_tweet[i]
        if data_tweet[i-1] not in ['ga', 'tidak', 'kurang', 'gak', 'enggak', 'nggak', 'tak']:
            negasi_data_tweet.append(word)
        else:
            word = "%s_%s" % (data_tweet[i-1], word)
            negasi_data_tweet.append(word)
    return negasi_data_tweet

