import streamlit as st

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgbm
from sklearn.preprocessing import OrdinalEncoder

from helper import _sections, _feature_columns, time_process, text_len

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.language import Language
from spacy.tokens import Token

from gensim.models import Phrases, LdaModel
from gensim.corpora import Dictionary

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

nlp = get_spacy_nlp()
bigrams, trigrams, dictionary = get_gensim_phrases_and_dictionary()
lda_model = get_lda_model()
section_encoder = get_section_encoder()
likes_regressor, retweets_regressor, replies_regressor = get_tree_ensembles()

tab1, tab2 = st.tabs(['Popularity Prediction', 'Visualizations'])

with tab1:
    with st.form('articleForm', clear_on_submit=False):

        titleText = st.text_input('Article Title', 'Input title/headline.')
        deckText = st.text_input('Article Subheading', 'Input subheading/drop head/deck that summarizes the article.')
        articleText = st.text_area('Article Text', 'Input the body of the news article.')
        news_section = st.selectbox('Section/Category', sorted(_sections))

        st.markdown("""---""")

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

        article_check1, article_check2, article_check3 = st.columns(3)
        with article_check1:
            articleHasVideo = st.checkbox('Article includes video?')
        with article_check2:
            articleHasAudio = st.checkbox('Article includes audio?')
        with article_check3:
            enableComments = st.checkbox('Enable reader comments?')

        st.markdown("""---""")

        tweetText = st.text_input('Tweet text', 'Input the Twitter tweet text for this article.')

        tweet_check1, tweet_check2 = st.columns(2)
        with tweet_check1:
            tweetHasVideo = st.checkbox('Tweet includes video?')
        with tweet_check2:
            tweetHasPhoto = st.checkbox('Tweet includes photograph?')

        st.markdown("""---""")

        submitted = st.form_submit_button('Submit')

    if submitted:
        doc = dictionary.doc2bow(trigrams[bigrams[[token.lemma_ for token in nlp(articleText) if not token._.is_excluded]]])
        topic_vector = np.array(lda_model.get_document_topics(doc, minimum_probability = -1))[:, 1]
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
        st.write(10**likes_score, 10**retweets_score, 10**replies_score)