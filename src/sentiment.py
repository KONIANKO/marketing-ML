import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from textblob import TextBlob
import torch.nn.functional as F
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import nltk
nltk.download('stopwords')

STOP_WORDS = stopwords.words('english')
FILE_PATH = '../data/reviews.txt'

MODEL_NAME = 'finiteautomata/bertweet-base-sentiment-analysis'
MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, normalization=True)
CLASSIFIER = pipeline('sentiment-analysis', model=MODEL, tokenizer=TOKENIZER)

def get_nouns(text):
    blob = TextBlob(text)
    return ' '.join([word for (word, tag) in blob.tags if tag in ["NN", "NNS", "NNP", "NNPS", "", ""]])

def get_adjectives(text):
    blob = TextBlob(text)
    return ' '.join([word for (word, tag) in blob.tags if tag in ["JJ", "JJR", "JJS"]])

def get_verbs(text):
    blob = TextBlob(text)
    return ' '.join([word for (word, tag) in blob.tags if tag in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]])

def preprocess_data(filepath: str) -> pd.core.frame.DataFrame:
    df = pd.read_csv(filepath, sep='\t', header=None)
    df = df.dropna()
    df.columns =['review']
    return df

def apply_model(sentence):

    batch = TOKENIZER(sentence, truncation=True, return_tensors='pt')
    
    with torch.no_grad():
        outputs = MODEL(**batch)
        predictions = F.softmax(outputs.logits, dim=1)
        labels = torch.argmax(predictions, dim=1)
        label = [MODEL.config.id2label[label_id] for label_id in labels.tolist()][0]
        score = torch.topk(predictions,1)[0].item()
    return [label, score]

def get_sentiment(sentence_list):
    sentiment = []
    for sentence in tqdm(sentence_list):
        result = apply_model(sentence)
        sentiment.append(result[0])
    return sentiment

def generate_positive_word_clouds(positive_df):
    wordcloud_pos_nouns = WordCloud(background_color = 'white', stopwords = STOP_WORDS, max_words = 20).generate(' '.join(positive_df['nouns']))
    plt.imshow(wordcloud_pos_nouns, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('../data/plots/nouns_positive_word_cloud.png', dpi=600)

    wordcloud_pos_adjectives = WordCloud(background_color = 'white', stopwords = STOP_WORDS, max_words = 20).generate(' '.join(positive_df['adjectives']))
    plt.imshow(wordcloud_pos_adjectives, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('../data/plots/adjectives_positive_word_cloud.png', dpi=600)

    wordcloud_pos_verbs = WordCloud(background_color = 'white', stopwords = STOP_WORDS, max_words = 20).generate(' '.join(positive_df['verbs']))
    plt.imshow(wordcloud_pos_verbs, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('../data/plots/verbs_positive_word_cloud.png', dpi=600)


def generate_negative_word_clouds(negative_df):
    wordcloud_neg_nouns = WordCloud(background_color = 'white', stopwords = STOP_WORDS, max_words = 20).generate(' '.join(negative_df['nouns']))
    plt.imshow(wordcloud_neg_nouns, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('../data/plots/nouns_negative_word_cloud.png', dpi=600)

    wordcloud_neg_adjectives = WordCloud(background_color = 'white', stopwords = STOP_WORDS, max_words = 20).generate(' '.join(negative_df['adjectives']))
    plt.imshow(wordcloud_neg_adjectives, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('../data/plots/adjectives_negative_word_cloud.png', dpi=600)

    wordcloud_neg_verbs = WordCloud(background_color = 'white', stopwords = STOP_WORDS, max_words = 20).generate(' '.join(negative_df['verbs']))
    plt.imshow(wordcloud_neg_verbs, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('../data/plots/verbs_negative_word_cloud.png', dpi=600)

def func(pct, allvalues):
    absolute = int(pct / 100.* np.sum(allvalues))
    return "{:.1f}%\n({:d})".format(pct, absolute)

def plot_sentiment_distripution(positive_df, neutral_df, negative_df):
    sentiments_classes = ['POSITIVE', 'NEUTRAL', 'NEGATIVE']
    values = [len(positive_df), len(neutral_df), len(negative_df)]
    explode = (0.1, 0.1, 0.1)
    colors = ( "green", "orange", "red")
    wp = { 'linewidth' : 1, 'edgecolor' : "green" }

    fig, ax = plt.subplots(figsize =(10, 7))
    wedges, texts, autotexts = ax.pie(values,
                                  autopct = lambda pct: func(pct, values),
                                  explode = explode,
                                  labels = sentiments_classes,
                                  shadow = True,
                                  colors = colors,
                                  startangle = 90,
                                  wedgeprops = wp,
                                  textprops = dict(color ="black"))
 
    ax.legend(wedges, sentiments_classes, title = "Sentiment", loc = "center left", bbox_to_anchor = (1, 0, 0.5, 1))
    plt.setp(autotexts, size = 8, weight ="bold")
    ax.set_title("Sentiment Distribution on Customer Reviews")
    plt.savefig('../data/plots/sentiment_distribution.png', dpi=600)

data = preprocess_data(FILE_PATH)
sentences = list(data['review'])
sentiment = get_sentiment(sentences)

data['sentiment']  = sentiment
data['nouns']      = data['review'].apply(get_nouns)
data['adjectives'] = data['review'].apply(get_adjectives)
data['verbs']      = data['review'].apply(get_verbs)

positive_df = data.loc[data['sentiment'] == 'POS']
neutral_df  = data.loc[data['sentiment'] == 'NEU']
negative_df = data.loc[data['sentiment'] == 'NEG']

generate_positive_word_clouds(positive_df)
generate_negative_word_clouds(negative_df)
plot_sentiment_distripution(positive_df, neutral_df, negative_df)
