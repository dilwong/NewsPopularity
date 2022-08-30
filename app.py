import streamlit as st

from typing import Union

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgbm
from sklearn.preprocessing import OrdinalEncoder

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns
import textwrap

from helper import _sections, _feature_columns, _topics, _select_categoricals, _conditional_proportions_options, time_process, text_len, percentile_rank, number_to_ordinal
from fake_news import fake_title, fake_deck, fake_article, fake_tweet, fake_section

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.language import Language
from spacy.tokens import Token

from gensim.models import Phrases, LdaModel
from gensim.corpora import Dictionary

if 'title_text' not in st.session_state:
    st.session_state['title_text'] = 'Input title/headline.'
if 'deck_text' not in st.session_state:
    st.session_state['deck_text'] = 'Input subheading/drop head/deck that summarizes the article.'
if 'article_text' not in st.session_state:
    st.session_state['article_text'] = 'Input the body of the news article.'
if 'tweet_text' not in st.session_state:
    st.session_state['tweet_text'] = 'Input the Twitter tweet text for this article.'

def insert_default_article():
    st.session_state['title_text'] = fake_title
    st.session_state['deck_text'] = fake_deck
    st.session_state['article_text'] = fake_article
    st.session_state['tweet_text'] = fake_tweet
    st.session_state['news_section'] = fake_section

@st.cache(hash_funcs={"spacy.lang.en.English": id})
def get_spacy_nlp():

    POS_blacklist = [
    'ADP',
    'ADV',
    'AUX',
    'CONJ',
    'CCONJ',
    'DET',
    'INJ',
    'PART',
    'PRON',
    'PUNCT',
    'SCONJ',
    ]

    stops = set(STOP_WORDS)
    stops.update(
        ["'s", "mr.", "mrs.", "ms.", "said", "according"]
    )

    @Language.component("lowercase_lemmas")
    def lowercase_lemmas(doc : spacy.tokens.doc.Doc) -> spacy.tokens.doc.Doc:
        for token in doc:
            token.lemma_ = token.lemma_.lower()
        return doc
        
    def get_is_excluded(token):
        return (token.pos_ in POS_blacklist) or (token.lemma_ in stops)

    if not Token.has_extension('is_excluded'):
        Token.set_extension('is_excluded', getter=get_is_excluded)

    nlp = spacy.load('en_core_web_sm', disable = ['ner'])
    nlp.add_pipe('lowercase_lemmas', last = True)

    return nlp

@st.cache(hash_funcs={"gensim.models.phrases.Phrases": id, "gensim.corpora.dictionary.Dictionary": id})
def get_gensim_phrases_and_dictionary():
    bigrams = Phrases.load('FinalModels/train_bigrams')
    trigrams = Phrases.load('FinalModels/train_trigrams')
    dictionary = Dictionary.load('FinalModels/train_dictionary')
    return bigrams, trigrams, dictionary

@st.cache(hash_funcs={"gensim.models.ldamodel.LdaModel": id})
def get_lda_model():
    return LdaModel.load(f'FinalModels/lda_model_130_29')

@st.cache(hash_funcs={"sklearn.preprocessing._encoders.OrdinalEncoder": id})
def get_section_encoder():
    return joblib.load('FinalModels/section_encoder.joblib')

@st.cache(hash_funcs={"lightgbm.basic.Booster": id})
def get_tree_ensembles():
    likes_regressor = lgbm.Booster(model_file = 'FinalModels/lgbm_likes.model')
    retweets_regressor = lgbm.Booster(model_file = 'FinalModels/lgbm_retweets.model')
    replies_regressor = lgbm.Booster(model_file = 'FinalModels/lgbm_replies.model')
    return likes_regressor, retweets_regressor, replies_regressor

@st.cache(hash_funcs={"pandas.core.frame.DataFrame": id})
def get_training_data():
    features_df = pd.read_parquet('ProcessedData/features.parquet.gzip')
    metrics_df = pd.read_parquet('ProcessedData/metrics.parquet.gzip')
    metadata_df = pd.read_parquet('ProcessedData/metadata.parquet.gzip')
    seconds_df = pd.read_parquet('ProcessedData/seconds.parquet.gzip')
    seconds_df['hour'] = seconds_df['seconds']//3600
    features_df = features_df.join(seconds_df['hour'].to_frame())
    return features_df, metrics_df, metadata_df

@st.cache(hash_funcs={"pandas.core.frame.DataFrame": id})
def get_visualization_data():
    curated_topics = []
    with open('ProcessedData/topic_keywords', 'rt') as f:
        for word in f:
            curated_topics.append(word.strip())
    keyword_loglikes_df = pd.read_parquet('ProcessedData/keyword_distributions.parquet.gzip')
    return curated_topics, keyword_loglikes_df

@st.cache(hash_funcs={"seaborn.axisgrid.FacetGrid": id})
def plot_conditional_proportions(column, features_frame, metrics_frame, xlabel = None, order = None, xticks = None, aspect = 1):
    with sns.plotting_context("notebook", font_scale = 1.5):
        reduced_frame = features_frame[column].to_frame()
        # There's certainly a more efficient (broadcasting-based) way to do this:
        likes_ptiles = metrics_frame['log_likes'].apply(lambda x: 100 * percentile_rank(x, metrics_frame['log_likes']))
        reduced_frame['Likes Percentile'] = pd.cut(likes_ptiles, [0, 20, 40, 60, 80, 100]).cat.rename_categories(lambda x: f'{x.left} - {x.right}')
        facetgrid = (reduced_frame
        .groupby(column)['Likes Percentile'].value_counts(normalize = True).rename('Conditional Proportion')
        .reset_index().rename(columns = {'level_1': 'Likes Percentile'})
        .pipe((sns.catplot, 'data'), kind = 'bar', hue = 'Likes Percentile', y = 'Conditional Proportion', x = column, order = order, height = 5, aspect = aspect, legend_out = True)
        )
        facetgrid.ax.set_ylabel('Relative Proportion')
        if xlabel is not None:
            facetgrid.ax.set_xlabel(xlabel)
        if xticks is not None:
            facetgrid.ax.set_xticklabels(xticks)
    return facetgrid

nlp = get_spacy_nlp()
bigrams, trigrams, dictionary = get_gensim_phrases_and_dictionary()
lda_model = get_lda_model()
section_encoder = get_section_encoder()
likes_regressor, retweets_regressor, replies_regressor = get_tree_ensembles()
features_df, metrics_df, metadata_df = get_training_data()
curated_topics, keyword_loglikes_df = get_visualization_data()

def get_document_similarities(vec : np.ndarray , metric : str = 'cosine', sort : Union[bool, str] = False) -> pd.DataFrame:
    r"""
    Given a vector of topic probabilities, return a DataFrame of similarities with respect to a corpus of news articles (the training data).

    Args:
        vec : np.ndarray
            The vector of topic probabilities.
        metric : str
            Either 'cosine' or 'hellinger' (case-insensitive).
            'both' returns a DataFrame with both cosine and Hellinger similarities, scaled between 0 and 1.
        sort : bool | str
            If not False, sort descending by similarity.
            If metric == 'both', provide a string 'cosine' or 'hellinger' (case-sensitive) to sort by cosine or Hellinger similarity. Otherwise ignored.
    
    Return:
        A pandas DataFrame containing values between [0, 1] that measure the similarity of the article described by vec and the articles in the training data.
        Values closer to 1 represent more similar articles, while values closer to 0 represent more dissimilar articles.
    """
    corpus_topics = features_df[_topics]
    if metric.lower() == 'cosine':
        l2_normalized_vec = vec/((vec**2).sum()**0.5)
        l2_normalized_corpus_topics = corpus_topics.divide((corpus_topics**2).sum(axis = 1)**0.5, axis = 'index')
        cos_similarity = (l2_normalized_corpus_topics * l2_normalized_vec).sum(axis = 1)
        cos_similarity = cos_similarity.rename('similarity') # Actually, since all elements of vec >= 0, cosine similarity already is [0, 1]
        if sort:
            cos_similarity = cos_similarity.sort_values(ascending = False)
        return cos_similarity.to_frame()
    if metric.lower() == 'hellinger':
        distance = ((0.5 * ((corpus_topics**0.5 - vec**0.5)**2).sum(axis = 1))**0.5)
        hellinger_similarity = (-distance + 1).rename('similarity') # most similar is distance = 0, least similar is distance = 1
        if sort:
            hellinger_similarity = hellinger_similarity.sort_values(ascending = False)
        return hellinger_similarity.to_frame()
    if metric.lower() == 'both':
        cos_similarity = get_document_similarities(vec, metric = 'cosine').rename(columns = {'similarity': 'cosine similarity'})
        hellinger_similarity = get_document_similarities(vec, metric = 'hellinger').rename(columns = {'similarity': 'hellinger similarity'})
        combined_similarity = cos_similarity.join(hellinger_similarity)
        if sort == 'cosine':
            combined_similarity = combined_similarity.sort_values(by = 'cosine similarity', ascending = False)
        elif sort == 'hellinger':
            combined_similarity = combined_similarity.sort_values(by = 'hellinger similarity', ascending = False)
        return combined_similarity
    assert False, 'metric must be cosine or hellinger'

def filter_articles_on_similarity(topic_sim : pd.DataFrame) -> pd.DataFrame:
    r"""
    Returns pandas DataFrame with at least 6 articles with large similarities.
    """
    MIN_ARTICLES = 6
    similarity_discount_factor = 0.8
    max_cosine = np.max(topic_sim['cosine similarity'])
    max_hellinger = np.max(topic_sim['hellinger similarity'])
    while True:
        topic_sim_mask = (topic_sim['cosine similarity'] > similarity_discount_factor * max_cosine) & (topic_sim['hellinger similarity'] > similarity_discount_factor * max_hellinger)
        if topic_sim_mask.sum() >= MIN_ARTICLES:
            break
        similarity_discount_factor -= 0.1
    return topic_sim[topic_sim_mask]

def print_articles(articles : pd.DataFrame):
    r'''
    Iterating through each row in a pandas DataFrame, write a link to each article (with the article title as the text) to the Streamlit app.
    If the article has been tagged by the New York Times, also write the tags.
    '''
    for _, row in articles.iterrows():
        if row['metadata_keywords'] is None:
            metadata_keywords = ''
        else:
            # HTML tags <nobr></nobr> cause keyword tags to overflow the expander box when they are too long compared to the screen size (e.g. rendered on mobile browsers)
            metadata_keywords = f"  \n<b>Tags</b>: {', '.join([f'<nobr>`{s}`</nobr>' for s in row['metadata_keywords'].split('|')])}"
        st.markdown(f"<b>[{row['title']}]({row['url']})</b>{metadata_keywords}", unsafe_allow_html = True)

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html = True)

tab1, tab2 = st.tabs(['Popularity Prediction', 'Visualizations'])

with tab1:
    with st.form('articleForm', clear_on_submit=False):

        st.subheader('Article Information')

        titleText = st.text_input('Article Title', key = 'title_text')
        deckText = st.text_input('Article Subheading', key = 'deck_text')
        articleText = st.text_area('Article Text', key = 'article_text')
        news_section = st.selectbox('Section/Category', sorted(_sections), key = 'news_section')

        article_check1, article_check2, article_check3 = st.columns(3)
        with article_check1:
            articleHasVideo = st.checkbox('Article includes video?')
        with article_check2:
            articleHasAudio = st.checkbox('Article includes audio?')
        with article_check3:
            enableComments = st.checkbox('Enable reader comments?')

        st.markdown("""---""")

        st.subheader('Tweet Information')

        tweetText = st.text_input('Tweet text', key = 'tweet_text')

        dateInput = st.date_input('Date')

        timecol1, timecol2, timecol3, timecol4 = st.columns(4)
        with timecol1:
            hour = st.selectbox('Hour', (f'{h:02}' for h in range(1, 12 + 1)))
        with timecol2:
            minute = st.selectbox('Minute', (f'{h:02}' for h in range(0, 60)))
        with timecol3:
            am_or_pm = st.selectbox('AM/PM', ['AM', 'PM'])
        with timecol4:
            timezone = st.selectbox('Timezone', ['US/Eastern', 'US/Central', 'US/Mountain', 'US/Pacific'])

        tweet_check1, tweet_check2 = st.columns(2)
        with tweet_check1:
            tweetHasVideo = st.checkbox('Tweet includes video?')
        with tweet_check2:
            tweetHasPhoto = st.checkbox('Tweet includes photograph?')

        st.markdown("""---""")

        submitted = st.form_submit_button('Submit')

    if not submitted:
        _ = st.button("I don't have a news article. Please fill in a default news article for me.", on_click = insert_default_article)
    else:
        doc = dictionary.doc2bow(trigrams[bigrams[[token.lemma_ for token in nlp(articleText) if not token._.is_excluded]]])
        lda_model.gamma_threshold = 1e-6
        lda_model.random_state = np.random.mtrand.RandomState(137)
        doc_topic = lda_model.get_document_topics(doc, minimum_probability = -1)
        topic_vector = np.array(doc_topic)[:, 1]
        topic_dict = {f'topic_{topic_index:03}' : topic_value for topic_index, topic_value in enumerate(topic_vector)}
        
        section_dict = {'section' : section_encoder.transform([[news_section]])[0, 0]}
        time_dict = time_process(dateInput, hour, minute, am_or_pm, timezone)
        bool_dict = {'tweet_has_video' : int(tweetHasVideo),
                    'tweet_has_photo' : int(tweetHasPhoto),
                    'article_has_video' : int(articleHasVideo),
                    'article_has_audio' : int(articleHasAudio),
                    'comments' : int(enableComments)
                    }
        len_dict = {'tweetlength' : text_len(tweetText),
                    'titlelength' : text_len(titleText),
                    'summarylength' : text_len(deckText),
                    'articlelength' : text_len(articleText)
                    }
        if len_dict['tweetlength'] is None: # Impute the number of words in the tweet if the tweet is not supplied.
            len_dict['tweetlength'] = 36
        
        features_dict = dict(
            **bool_dict,
            **len_dict,
            **section_dict,
            **time_dict,
            **topic_dict
        )
        features = pd.DataFrame([features_dict])[_feature_columns].to_numpy()
        
        likes_score = float(likes_regressor.predict(features))
        retweets_score = float(retweets_regressor.predict(features))
        replies_score = float(replies_regressor.predict(features))
        likes_percentile = int(percentile_rank(likes_score, metrics_df['log_likes']) * 100)
        retweets_percentile = int(percentile_rank(retweets_score, metrics_df['log_retweets']) * 100)
        replies_percentile = int(percentile_rank(replies_score, metrics_df['log_replies']) * 100)

        topic_sim = get_document_similarities(topic_vector, metric = 'both').join(metadata_df).join(metrics_df)
        topic_sim = topic_sim.drop_duplicates(subset = 'title') # Turns out that the NYT reposts its articles on Twitter multiple times, sometimes with wildly differing social media stats.
        topic_sim = filter_articles_on_similarity(topic_sim).sort_values(by = 'log_likes', ascending = False)
        most_popular_articles = topic_sim.iloc[:3, :]
        least_popular_articles = topic_sim.iloc[-1:-3-1:-1, :]
        
        with st.expander('Prediction Results', expanded = True):
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric('Likes Percentile', value = number_to_ordinal(likes_percentile))
            with metric_col2:
                st.metric('Retweets Percentile', value = number_to_ordinal(retweets_percentile))
            with metric_col3:
                st.metric('Replies Percentile', value = number_to_ordinal(replies_percentile))
            st.write(f'Your article has predicted numbers of likes, retweets, and replies in the {number_to_ordinal(likes_percentile)}, {number_to_ordinal(retweets_percentile)}, and {number_to_ordinal(replies_percentile)} percentiles.')
            
            st.markdown('#### Popular related articles')
            print_articles(most_popular_articles)

            st.markdown('#### Unpopular related articles')
            print_articles(least_popular_articles)

with tab2:
    st.subheader('Distribution Plot of Twitter Likes per Keyword')
    order = st.multiselect('Keywords/Tags:', curated_topics)
    if order:
        if len(order) > 10:
            st.write('Please input 10 or less keywords.')
        else:
            if len(order) > 5:
                st.write("Please don't exceed 5 keywords.")
            fig, ax = plt.subplots(figsize = (25, 5))
            ax.grid(zorder = 0)
            ax.yaxis.set_major_locator(MultipleLocator(1))
            boxen = sns.boxenplot(x = 'keyword', y = 'log_likes', data = keyword_loglikes_df, ax = ax, order = order)
            ax.set_xticklabels([textwrap.fill(lab, 6, break_long_words = False) for lab in order])
            plt.setp(boxen.collections, zorder = 10)
            plt.setp(boxen.lines, zorder=11)
            ax.set_ylim(1,5)
            ax.set_xlabel('Keyword Tags', fontsize = 22)
            ax.set_ylabel(r'$\log_{10}({\rm Likes})$', fontsize = 22)
            ax.tick_params(axis="y", labelsize=18)
            ax.tick_params(axis="x", labelsize=18)
            st.pyplot(fig = fig)
    
    st.subheader('Likes Percentile Proportion Conditional on Feature')
    barplot_feature = st.selectbox('Feature:', [''] + list(_select_categoricals.keys()))
    # Too bad pattern matching isn't available until Python 3.10
    if barplot_feature != '':
        relative_props_plot = plot_conditional_proportions(
            _select_categoricals[barplot_feature],
            features_frame = features_df,
            metrics_frame = metrics_df,
            **_conditional_proportions_options[_select_categoricals[barplot_feature]]
        )
        # st.pyplot(relative_props_plot.fig)
        st.pyplot(relative_props_plot)