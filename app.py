# Import Framework Flask, werkzeug, import file utils_process, import file model
from flask import Flask, render_template, request, session
from werkzeug.utils import secure_filename
from utils_process import text_preprocessing
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc

# Perpustakaan untuk Manipulasi Data
import pandas as pd
import csv, json, os
import numpy as np
import pickle
from collections import Counter
from itertools import cycle

# Perpustakaan untuk visualisasi
import base64
from io import BytesIO
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
from wordcloud import WordCloud, STOPWORDS
import plotly
import plotly.figure_factory as ff 
import plotly.express as px
import plotly.graph_objects as go 
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE

# Perpustakaan untuk Abaikan Warning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Perpustakaan untuk generate secret key
import uuid
secret_key = uuid.uuid4().hex
print(secret_key)


# Deklarasi aplikasi Flask
app = Flask(__name__)
app.static_folder = 'static'

# Tentukan kunci rahasia untuk mengaktifkan sesi
app.secret_key = '8da96fb9a89d456aaf47f825eda11429'

# Tentukan folder untuk menyimpan file yang diunggah untuk diproses lebih lanjut
UPLOAD_FOLDER = os.path.join(app.instance_path, 'uploads')

# Konfigurasikan upload file path flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load dataset final
data = pd.read_csv("dataset/Final Data.csv")

# Load dataset hasil preprocessing
data_preprocessing = pd.read_csv("dataset/Hasil Preprocessing.csv")


# Load model 
model_filename = 'model/svm-bert-model.pkl'
with open(model_filename, 'rb') as file:
    loaded_svm_model = pickle.load(file)

# Variabel global untuk menyimpan hasil data, gambar, dan grafik dalam kondisi global
global new_results_dict
global fig_sentiment_by_date
global fig_sentiment_pie
global fig_sentiment_bar
global fig_top_25_words_neutral
global fig_top_25_words_negative
global fig_top_25_words_positive
global vis_negative
global vis_neutral
global vis_positive
global total_tweet
global total_reply_count
global total_retweet_count
global total_user

# Function untuk visualisasi sentimen berdasarkan tanggal
def viz_sentiment_by_date(data, column, width):
    # Konversi kolom 'created_at' ke tipe datetime
    data['created_at'] = pd.to_datetime(data['created_at'])

    # Menghitung jumlah ulasan per tanggal
    review_count_by_date = data.groupby(data['created_at'].dt.date)['id_str'].count().reset_index()
    review_count_by_date.columns = ['Date', 'Review Count']

    # Menghitung jumlah sentimen positif, negatif, dan netral per tanggal
    sentiment_count_by_date = data.groupby([data['created_at'].dt.date, column])['id_str'].count().reset_index()
    sentiment_count_by_date.columns = ['Date', 'Sentiment', 'Sentiment Count']

    # Membuat grafik area dengan Plotly go Figure
    fig = go.Figure()

    # Menambahkan data untuk setiap sentimen
    for sentiment in sentiment_count_by_date['Sentiment'].unique():
        sentiment_data = sentiment_count_by_date[sentiment_count_by_date['Sentiment'] == sentiment]
        # Tambahkan area plot
        fig.add_trace(go.Scatter(x=sentiment_data['Date'], y=sentiment_data['Sentiment Count'],
                                 mode='lines', name=sentiment, stackgroup='one',
                                 line=dict(width=0.5),
                                 fill='tozeroy',  # Menambahkan area di bawah garis
                                 fillcolor={'Positif': 'rgba(100, 149, 237, 0.3)', 'Negatif': 'rgba(222, 49, 99, 0.3)', 'Netral': 'rgba(64, 224, 208, 0.3)'}[sentiment]))

    # Mengatur label sumbu x dan y
    fig.update_xaxes(title_text='Tanggal')
    fig.update_yaxes(title_text='Jumlah Data')

    # Set properti hoverlabel_font_color agar teks pada hover info berwarna putih
    fig.update_traces(hoverlabel_font_color='black')

    # Membuat slider untuk periode waktu
    fig.update_layout(
        xaxis=dict(rangeslider=dict(visible=True), type="date"),
    )

    # Mengatur template dan warna latar belakang
    fig.update_layout(template='plotly_white')
    fig.update_layout({
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)'
    })

    # Mengatur warna grid
    fig.update_xaxes(color='black', showgrid=False)
    fig.update_yaxes(color='black', showgrid=False)

    # Membuat grafik responsif
    fig.update_layout(
        autosize=True,  # Mengaktifkan autosize
        margin=dict(l=100, r=0, b=50, t=0),  # Menghilangkan margin
        legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="center", x=0.5),  # Menyesuaikan posisi legenda
        font=dict(size=14)  # Menyesuaikan ukuran font
    )

    # Mengatur ukuran grafik
    fig.update_layout(width=width)
    return fig



# Function untuk visualisasi sentimen dalam bentuk grafik lingkaran
def viz_sentiment_pie(data, column):
    # Hitung jumlah data berdasarkan kolom 'sentiment'
    sizes = data[column].value_counts().values
    labels = ['Positif', 'Netral', 'Negatif']
    
    # Membuat grafik area dengan Plotly px
    fig = px.pie(data, names=labels, values=sizes, hole=0.6,
                 color_discrete_sequence=px.colors.qualitative.T10,
                 width = 400)
    
    # Mengatur tampilan huruf dan legend
    fig.update_traces(textfont=dict(color='#000'))
    fig.update_layout(legend_title_font=dict(color='black'))
    fig.update_layout(title_font=dict(color='black'))

    # Mengatur template dan warna latar belakang
    fig.update_layout(legend_font_color='black')
    fig.update_layout({
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)'
    })

    # Mengatur warna grid
    fig.update_xaxes(color='black', showgrid=False) 
    fig.update_yaxes(color='black', showgrid=False) 

    # Membuat grafik responsif
    fig.update_layout(
        autosize=True,  # Mengaktifkan autosize
        margin=dict(l=50, r=60, b=0, t=0),  # Menghilangkan margin
        legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="center", x=0.5),  # Menyesuaikan posisi legenda
        font=dict(size=14)  # Menyesuaikan ukuran font
    )
    return fig

# Function untuk visualisasi sentimen dalam bentuk grafik batang
def viz_sentiment_bar(data, column):
    # Hitung jumlah data berdasarkan kolom 'sentiment'
    sizes = data[column].value_counts()
    labels = ['Positif', 'Netral', 'Negatif']

    colors = {'Negatif': '#DE3163', 'Positif': '#6495ED', 'Netral': '#40E0D0'}

    # Buat plot grafik batang dengan Plotly
    fig = go.Figure()

    # Tambahkan bar untuk setiap sentimen
    for label in labels:
        fig.add_trace(go.Bar(x=[label], y=[sizes[label]], name=label,
                             marker=dict(color=colors[label]),
                             text=[sizes[label]], textposition='outside',
                             textfont=dict(color='black')))

    # Atur tampilan legenda
    fig.update_layout(showlegend=True)
    
    # Set properti hoverlabel_font_color agar teks pada hover info berwarna putih
    fig.update_traces(hoverlabel_font_color='black')
    
    # Atur hovertemplate untuk hanya menampilkan Sentimen dan Jumlah data
    fig.update_traces(hovertemplate='Sentiment: %{x}<br>Jumlah Data: %{y}')
    
    # Mengatur label sumbu x dan y
    fig.update_xaxes(title_text='Sentiment')
    fig.update_yaxes(title_text='Jumlah Data')
    
    # Mengatur template dan warna latar belakang
    fig.update_layout(template='plotly_white')
    fig.update_layout({
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)'
    })

    # Mengatur warna grid
    fig.update_xaxes(color='black', showgrid=False)
    fig.update_yaxes(color='black', showgrid=False)
    
    # Membuat grafik responsif
    fig.update_layout(
        autosize=True,  # Mengaktifkan autosize
        margin=dict(l=50, r=50, b=0, t=0),  # Menghilangkan margin
        legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="center", x=0.5),  # Menyesuaikan posisi legenda
        font=dict(size=14)  # Menyesuaikan ukuran font
    )
    # Mengatur ukuran grafik
    fig.update_layout(width=400)
    return fig


# Function untuk visualisasi top words group masing-masing sentimen 
def viz_top_word_group(data):
    # Hitung frekuensi kata-kata dalam masing-masing sentimen
    positive_words = " ".join(data[data['sentiment'] == 'Positif']['full_text']).split()
    negative_words = " ".join(data[data['sentiment'] == 'Negatif']['full_text']).split()
    neutral_words = " ".join(data[data['sentiment'] == 'Netral']['full_text']).split()

    positive_word_counts = pd.Series(positive_words).value_counts()
    negative_word_counts = pd.Series(negative_words).value_counts()
    neutral_word_counts = pd.Series(neutral_words).value_counts()

    # Buat grafik bar dengan Plotly
    fig = go.Figure(data=[
        go.Bar(x=positive_word_counts.index[:25], y=positive_word_counts.values[:25], name='Positif', marker_color='#6495ED'),
        go.Bar(x=negative_word_counts.index[:25], y=negative_word_counts.values[:25], name='Negatif', marker_color='#DE3163'),
        go.Bar(x=neutral_word_counts.index[:25], y=neutral_word_counts.values[:25], name='Netral', marker_color='#40E0D0')
    ])

    # Mengatur label sumbu x dan y
    fig.update_xaxes(title="Kata")
    fig.update_yaxes(title="Frekuensi")

    # Set properti hoverlabel_font_color agar teks pada hover info berwarna putih
    fig.update_traces(hoverlabel_font_color='black')

    # Mengatur template dan warna latar belakang
    fig.update_layout(barmode='group', template='plotly_white')
    fig.update_layout({
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)'
    })

    # Mengatur warna grid
    fig.update_xaxes(color='black', showgrid=False)
    fig.update_yaxes(color='black', showgrid=False)

    # Mengatur ukuran grafik
    fig.update_layout(width=1200, height=500)

    # Membuat grafik responsif
    fig.update_layout(
        autosize=True,  # Mengaktifkan autosize
        margin=dict(l=120, r=50, b=0, t=0),  # Menghilangkan margin
        legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="center", x=0.5),  # Menyesuaikan posisi legenda
        font=dict(size=14)  # Menyesuaikan ukuran font
    )
    return fig


# Function untuk visualisasi top words masing-masing sentimen
def visualize_top_words(data, sentiment_value, num_top_words=10):
    count_data = data[data['prediksi_sentiment'] == sentiment_value]

    def get_top_words(data, num_top_words):
        word_count = pd.Series(' '.join(data['text_clean']).split()).value_counts()
        return word_count.head(num_top_words)
    
    top_words = get_top_words(count_data, num_top_words)

    custom_colors = ['#FF5733', '#33FF57', '#5733FF', '#FFFF33', '#33FFFF', '#FF33FF', '#FF5733', '#33FF57', '#5733FF', '#FFFF33',
                    '#33FFFF', '#FF33FF', '#FF5733', '#33FF57', '#5733FF', '#FFFF33', '#33FFFF', '#FF33FF', '#FF5733', '#33FF57',
                    '#5733FF', '#FFFF33', '#33FFFF', '#FF33FF']

    # Buat grafik bar dengan Plotly
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=top_words.index,  
        x=top_words.values,  
        orientation='h',
        name= str(sentiment_value),
        marker_color=custom_colors[:len(top_words)]
    ))

    # Membuat grafik responsif
    fig.update_layout(
        xaxis_title='Count',  
        yaxis_title='Words',  
        width=400,  
        barmode='group',
        margin=dict(l=0, r=0, b=0, t=0),
        font=dict(size=14)
    )

    # Mengatur tampilan huruf dan legend
    fig.update_layout(title_font=dict(color='black'))
    fig.update_layout(showlegend=False)

    # Mengatur template dan warna latar belakang
    fig.update_layout(template='plotly_white')
    fig.update_layout({
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)'
    })

    # Mengatur warna grid
    fig.update_xaxes(color='black', showgrid=False) 
    fig.update_yaxes(color='black', showgrid=False) 

    # Mengatur ukuran grafik
    fig.update_layout(width=800)
    return fig



# Function untuk menampilkan visualisasi Wordcloud
def render_word_cloud(corpus):
    '''Generates a word cloud using all the words in the corpus.
    '''
    wordcloud = WordCloud(mode = "RGBA", width=1600, height=1000,max_font_size=200, background_color=None, colormap = 'magma', stopwords=STOPWORDS).generate(corpus)
    fig_file = BytesIO()
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(fig_file, format='png', dpi=600, bbox_inches = 'tight', pad_inches = 0)
    fig_file.seek(0)
    fig_data_png = fig_file.getvalue()
    result = base64.b64encode(fig_data_png)
    return result.decode('utf-8')


# Function untuk menampilkan visualisasi Confusion Matrix dan Classification Report
def plotConfusionMatrixAndReport(confusion_matrix_file, classification_report_file):
    # Load confusion matrix dan classification report dari CSV files
    cm_df = pd.read_csv(confusion_matrix_file, index_col=0)
    report_df = pd.read_csv(classification_report_file, index_col=0)
    
    # Plot confusion matrix menggunakan Plotly
    fig_conf_matrix = go.Figure(data=go.Heatmap(
        z=cm_df.values,
        x=cm_df.columns,
        y=cm_df.index,
        text=cm_df.values,  
        hoverinfo='text',  
        colorscale='Blues',
        reversescale=True,
    ))

    for i in range(len(cm_df.index)):
        for j in range(len(cm_df.columns)):
            fig_conf_matrix.add_annotation(
                x=cm_df.columns[j],
                y=cm_df.index[i],
                text=str(cm_df.iloc[i, j]),
                showarrow=False,
                font=dict(color='white' if cm_df.iloc[i, j] < cm_df.values.max() else 'black')
            )

    # Membuat grafik responsif
    fig_conf_matrix.update_layout(
        xaxis_title='Predicted',
        yaxis_title='Actual',
        width=600,
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
    )

    # Mengatur template dan warna latar belakang
    fig_conf_matrix.update_layout({
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)'
    })

    # Mengatur warna label sumbu x dan y
    fig_conf_matrix.update_xaxes(color='black') 
    fig_conf_matrix.update_yaxes(color='black') 

    # Format classification report values to two decimal places
    report_df = report_df.round(2)

    # Rename the index column to an empty string
    report_df.index.name = ''
    
    # Create a table for classification report using Plotly
    fig_report = ff.create_table(report_df.reset_index())

    # Membuat grafik responsif
    fig_report.update_layout(
        width=600,
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
    )

    # Mengatur template dan warna latar belakang
    fig_report.update_layout({
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)'
    })

    return fig_conf_matrix, fig_report


# Function untuk menampilkan visualisasi Word Embedding
def word_embeddings(data):
    # Ekstrak embeddings dan sentiments
    embeddings = np.array(data["embeddings"].tolist())  # Konvert list menjadi NumPy array
    sentiments = list(data["sentiment"])

    # Perform t-SNE
    tsne = TSNE(n_components=3, random_state=42)
    embedding_tsne = tsne.fit_transform(embeddings)

    # Membuat DataFrame untuk t-SNE data
    tsne_data_3d = {"X": embedding_tsne[:, 0], "Y": embedding_tsne[:, 1], "Z": embedding_tsne[:, 2], "Sentiment": sentiments}
    tsne_df_3d = pd.DataFrame(tsne_data_3d)

    # Plot t-SNE menggunakan Plotly 
    fig = px.scatter_3d(tsne_df_3d, x="X", y="Y", z="Z", color="Sentiment", hover_data=["X", "Y", "Z", "Sentiment"])
    
    # Mengatur template dan warna latar belakang
    fig.update_layout(template='plotly_white')
    fig.update_layout({
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)'
    })

    # Mengatur template dan warna latar belakang
    fig.update_xaxes(color='black') 
    fig.update_yaxes(color='black') 

    # Mengatur ukuran grafik
    fig.update_layout(width=1200)
    return fig


def plot_roc_curve(df_roc):
    # Assuming df_roc is the DataFrame containing the AUC, FPR, and TPR values
    n_classes = len(df_roc) - 2  # Exclude 'Micro-average' and 'Macro-average'

    # Define the line width and colors
    lw = 2
    colors = ['aqua', 'darkorange', 'cornflowerblue']

    # Create a Plotly figure
    fig = go.Figure()

    # Plot ROC curves for each class
    for i in range(n_classes):
        class_name = df_roc.loc[i, 'Class']
        fpr = df_roc.loc[i, 'FPR']
        tpr = df_roc.loc[i, 'TPR']
        auc_value = df_roc.loc[i, 'AUC']
        
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                                name=f'{class_name} (AUC={auc_value:.2f})', 
                                line=dict(width=lw, color=colors[i])))

    # Plot Micro-average
    micro_fpr = df_roc.loc[df_roc['Class'] == 'Micro-average', 'FPR'].values[0]
    micro_tpr = df_roc.loc[df_roc['Class'] == 'Micro-average', 'TPR'].values[0]
    micro_auc_value = df_roc.loc[df_roc['Class'] == 'Micro-average', 'AUC'].values[0]
    fig.add_trace(go.Scatter(x=micro_fpr, y=micro_tpr, mode='lines', 
                            name=f'Micro-average (AUC={micro_auc_value:.2f})', 
                            line=dict(width=lw, color='deeppink', dash='dash')))

    # Plot Macro-average
    macro_fpr = df_roc.loc[df_roc['Class'] == 'Macro-average', 'FPR'].values[0]
    macro_tpr = df_roc.loc[df_roc['Class'] == 'Macro-average', 'TPR'].values[0]
    macro_auc_value = df_roc.loc[df_roc['Class'] == 'Macro-average', 'AUC'].values[0]
    fig.add_trace(go.Scatter(x=macro_fpr, y=macro_tpr, mode='lines', 
                            name=f'Macro-average (AUC={macro_auc_value:.2f})', 
                            line=dict(width=lw, color='green', dash='dash')))

    # Update layout
    fig.update_layout(title='ROC Curve for Sentiment Analysis',
                    xaxis=dict(title='False Positive Rate'),
                    yaxis=dict(title='True Positive Rate'),
                    legend=dict(x=0.98, y=0.02, traceorder='normal'),
                    width=600,
                    showlegend=True)

    # Update traces and layout for a dark theme
    fig.update_traces(textfont=dict(color='#fff'))
    fig.update_traces(hoverlabel_font_color='black')
    fig.update_layout(legend_title_font=dict(color='black'))
    fig.update_layout(template='plotly_white', title_font=dict(color='black'))
    fig.update_xaxes(color='black', showgrid=False)
    fig.update_yaxes(color='black', showgrid=False)
    return fig


# Load tokenizer dan model untuk word embedding
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
model = BertModel.from_pretrained("indobenchmark/indobert-base-p1")

# Function untuk tokenisasi dan vektorisasi teks menggunakan IndoBERT
def vectorize_text(text):
    tokens = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state.mean(dim=1).squeeze().numpy()


# Route untuk halaman awal
@app.route('/')
# Function untuk merender halaman index.html
def home_page():
    # Menghitung total tweet
    total_tweet = len(data)

    # Menghitung total retweet_count dan reply_count
    total_retweet_count = data["retweet_count"].sum()
    total_reply_count = data["reply_count"].sum()

    # Menghitung total user
    total_user = len(data["username"].unique())

    # Menampilkan plot sentimen berdasarkan tanggal
    fig_sentiment_by_date = viz_sentiment_by_date(data, column='sentiment', width=800)
    fig_sentiment_by_date = json.dumps(fig_sentiment_by_date, cls=plotly.utils.PlotlyJSONEncoder)

    # Menampilkan plot pie sentimen
    fig_sentiment_pie = viz_sentiment_pie(data, column='sentiment')
    fig_sentiment_pie = json.dumps(fig_sentiment_pie, cls=plotly.utils.PlotlyJSONEncoder)

    # Menampilkan plot bar sentimen
    fig_sentiment_bar = viz_sentiment_bar(data, column='sentiment')
    fig_sentiment_bar = json.dumps(fig_sentiment_bar, cls=plotly.utils.PlotlyJSONEncoder)

    # Menampilkan top 25 words masing-masing sentimen
    fig_top_words_group = viz_top_word_group(data)
    fig_top_words_group = json.dumps(fig_top_words_group, cls=plotly.utils.PlotlyJSONEncoder)
    
    # # Menampilkan hasil evaluasi confusion matrix dan classification report
    confusion_matrix_file = 'dataset/results_confusion_matrix.csv'
    classification_report_file = 'dataset/results_classification_report.csv'
    roc_file = 'dataset/ROC-AUC.csv'

    fig_cm, fig_report = plotConfusionMatrixAndReport(confusion_matrix_file, classification_report_file)
    fig_cm = json.dumps(fig_cm, cls=plotly.utils.PlotlyJSONEncoder)
    fig_report = json.dumps(fig_report, cls=plotly.utils.PlotlyJSONEncoder)

    df_roc = pd.read_pickle('model/ROC-AUC.pkl')
    fig_roc_auc = plot_roc_curve(df_roc)
    fig_roc_auc = json.dumps(fig_roc_auc, cls=plotly.utils.PlotlyJSONEncoder)

    # Menampilkan tabel data hasil preprocessing
    preprocessing_dict = data_preprocessing.to_dict('records')

    # Load embedding file
    embeddings = pd.read_pickle('model/Hasil Embedding Train-Test.pkl')
    # sentiment_map = {0: 'Negatif', 1: 'Positif', 2: 'Netral'}
    # embeddings['sentiment'] = embeddings.sentiment.replace(sentiment_map)
    
    fig_word_embeddings = word_embeddings(embeddings)
    fig_word_embeddings = json.dumps(fig_word_embeddings, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Return render template yang mana akan merender halaman awal
    # Mengirim data dari variabel data, data grafik, data gambar ke sisi client
    return render_template('index.html',
                           total_tweet=total_tweet,
                           total_retweet_count=total_retweet_count,
                           total_reply_count=total_reply_count,
                           total_user=total_user,
                           fig_sentiment_by_date=fig_sentiment_by_date,
                           fig_sentiment_pie=fig_sentiment_pie,
                           fig_sentiment_bar=fig_sentiment_bar,
                           fig_top_words_group=fig_top_words_group, 
                           data=preprocessing_dict,
                           cm_viz=fig_cm,
                           cr_viz=fig_report,
                           auc_viz=fig_roc_auc,
                           fig_word_embeddings=fig_word_embeddings)


# Route untuk halaman prediction file yang menjalankan method GET dan POST
# POST mengirim data yang diupload user ke bagian sistem (Backend) untuk diolah
# Data yang telah diolah oleh sistem yang kemudian dikirimkan kembali ke sisi client (Frontend) untuk ditampilkan
@app.route('/prediction', methods=['GET','POST'])
def prediction_file():
    # Variabel global untuk menyimpan hasil data, gambar, dan grafik dalam kondisi global
    global new_results_dict
    global fig_sentiment_by_date
    global fig_sentiment_pie
    global fig_sentiment_bar
    global fig_top_25_words_neutral
    global fig_top_25_words_negative
    global fig_top_25_words_positive
    global vis_negative
    global vis_neutral
    global vis_positive
    global total_tweet
    global total_reply_count
    global total_retweet_count
    global total_user
    
    css_file = "static\css\black-dashboard.css"

    # Periksa apakah file sudah diunggah
    if request.method == 'POST':
        # Dapatkan file yang diunggah
        uploaded_df = request.files['file']
        # Dapatkan nama file dan simpan file ke UPLOAD_FOLDER
        data_filename = secure_filename(uploaded_df.filename)
        # Simpan jalur file yang diunggah di sesi
        uploaded_df.save(os.path.join(app.config['UPLOAD_FOLDER'], data_filename))
        session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
        filepath = session['uploaded_data_file_path']

        # Muat data dari file CSV ke dalam variabel dict_data dengan format list
        dict_data = []
        with open(filepath, encoding="latin-1") as file:
            csv_file = csv.DictReader(file)
            for row in csv_file:
                dict_data.append(row)
        
        # Bersihkan data teks dan simpan hasilnya dalam variabel results dengan format list
        results = []
        for row in dict_data:
            row['text_clean'] = text_preprocessing(row['full_text']) 
            results.append(dict(row))   

        # Ubah daftar kamus menjadi Pandas DataFrame
        df_results = pd.DataFrame.from_dict(results)
        print(df_results)

        # Hanya mengambil kolom tertentu saja
        df_results = df_results[['created_at', 'id_str', 'full_text', 'text_clean', 'quote_count', 'reply_count',
                                'retweet_count', 'favorite_count', 'lang', 'username', 'tweet_url']]
        
        # Konversi kolom ke tipe data numerik jika diperlukan
        df_results["retweet_count"] = pd.to_numeric(df_results["retweet_count"], errors='coerce')
        df_results["reply_count"] = pd.to_numeric(df_results["reply_count"], errors='coerce')

        # Menghitung total tweet
        total_tweet = len(df_results)

        # Menghitung total retweet_count dan reply_count
        total_retweet_count = df_results["retweet_count"].sum()
        total_reply_count = df_results["reply_count"].sum()

        # Menghitung total user
        total_user = len(df_results["username"].unique())

        # Apply tokenization and vectorization to the text column
        df_results['text_vector'] = df_results['full_text'].apply(vectorize_text)

        # Convert NumPy array to list
        df_results['text_vector'] = df_results['text_vector'].apply(lambda x: x.tolist())

        # Predict using the SVM model
        predictions = loaded_svm_model.predict(df_results['text_vector'].tolist())

        # Buat kolom baru di DataFrame `test_data` untuk hasil sentimen
        df_results['prediksi_sentiment'] = predictions
        print(df_results['prediksi_sentiment'].unique())

        # Mengkonversi pandas dataframe menjadi dictionary
        new_results_dict = df_results.to_dict('records')

        ## --------------- Kode Untuk Menampilkan Visualisasi Dari Hasil Prediksi --------------- ##
        sentiment_map = {0: 'Negatif', 1: 'Positif', 2: 'Netral'}
        df_results['prediksi_sentiment'] = df_results.prediksi_sentiment.replace(sentiment_map)

        # Menampilkan plot pie sentimen
        fig_sentiment_pie = viz_sentiment_pie(df_results, column='prediksi_sentiment')
        fig_sentiment_pie = json.dumps(fig_sentiment_pie, cls=plotly.utils.PlotlyJSONEncoder)

        # Menampilkan plot bar sentimen
        fig_sentiment_bar = viz_sentiment_bar(df_results, column='prediksi_sentiment')
        fig_sentiment_bar = json.dumps(fig_sentiment_bar, cls=plotly.utils.PlotlyJSONEncoder)

        # Filter data berdasarkan hasil prediksi masing-masing sentimen  
        df_wc_pos = df_results[df_results['prediksi_sentiment'] == 'Positif']
        df_wc_neg = df_results[df_results['prediksi_sentiment'] == 'Negatif']
        df_wc_neu = df_results[df_results['prediksi_sentiment'] == 'Netral']

        # Menampilkan wordcloud sentiment positif 
        wc_pos = ' '.join(df_wc_pos['text_clean'])
        wc_neg = ' '.join(df_wc_neg['text_clean'])
        wc_neu = ' '.join(df_wc_neu['text_clean'])

        vis_positive = render_word_cloud(wc_pos)
        vis_negative = render_word_cloud(wc_neg)
        vis_neutral = render_word_cloud(wc_neu)

        # Menampilkan plot sentimen berdasarkan tanggal
        fig_sentiment_by_date = viz_sentiment_by_date(df_results, column='prediksi_sentiment', width=1200)
        fig_sentiment_by_date = json.dumps(fig_sentiment_by_date, cls=plotly.utils.PlotlyJSONEncoder)

        # Menampilkan top 25 words masing-masing sentimen
        fig_top_25_words_positive = visualize_top_words(df_results, 'Positif', num_top_words=25)
        fig_top_25_words_negative = visualize_top_words(df_results, 'Negatif', num_top_words=25)
        fig_top_25_words_neutral = visualize_top_words(df_results, 'Netral', num_top_words=25)
        fig_top_25_words_positive = json.dumps(fig_top_25_words_positive, cls=plotly.utils.PlotlyJSONEncoder)
        fig_top_25_words_negative = json.dumps(fig_top_25_words_negative, cls=plotly.utils.PlotlyJSONEncoder)
        fig_top_25_words_neutral = json.dumps(fig_top_25_words_neutral, cls=plotly.utils.PlotlyJSONEncoder)


    # Return render template yang mana akan merender halaman testing
    # Mengirim data dari variabel data, data grafik, data gambar ke sisi client
    return render_template('result.html', 
                           data=new_results_dict,
                           total_tweet=total_tweet,
                           total_retweet_count=total_retweet_count,
                           total_reply_count=total_reply_count,
                           total_user=total_user,
                           fig_sentiment_by_date=fig_sentiment_by_date,
                           fig_sentiment_pie=fig_sentiment_pie,
                           fig_sentiment_bar=fig_sentiment_bar,
                           fig_top_25_words_positive=fig_top_25_words_positive,
                           fig_top_25_words_negative=fig_top_25_words_negative,
                           fig_top_25_words_neutral=fig_top_25_words_neutral,
                           css_file=css_file,
                           vis_positive=vis_positive,
                           vis_negative=vis_negative,
                           vis_neutral=vis_neutral
                           )



# Menjalankan aplikasi flask
if __name__ == '__main__':
	app.run(debug=True)