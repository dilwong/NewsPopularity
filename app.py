import streamlit as st
import datetime

import numpy as np

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.language import Language
from spacy.tokens import Token

from gensim.models import Phrases, LdaModel
from gensim.corpora import Dictionary

_sections = ['world',
            'world/europe',
            'us',
            'us/politics',
            'nyregion',
            'business',
            'world/asia',
            'sports/olympics',
            'opinion',
            'movies',
            'sports',
            'arts/music',
            'arts/television',
            'technology',
            'health',
            'magazine',
            'science',
            'climate',
            'article',
            'style',
            'wirecutter',
            'espanol',
            'recipes',
            'world/middleeast',
            'business/economy',
            'podcasts/the-daily',
            'business/media',
            'world/americas',
            'arts',
            'sports/football',
            'travel',
            'world/africa',
            'sports/basketball',
            'world/australia',
            'realestate',
            'books',
            'arts/design',
            'sports/tennis',
            'well/live',
            'dining',
            'briefing',
            'upshot',
            'sports/soccer',
            'well/mind',
            'theater',
            'headway',
            'sports/baseball',
            'world/canada',
            'podcasts',
            'well',
            'books/review',
            'obituaries',
            'business/energy-environment',
            'well/move',
            'well/family',
            'sports/golf',
            'your-money',
            'well/eat',
            'arts/dance',
            'special-series',
            'fashion',
            'business/dealbook'
            ]

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

nlp = get_spacy_nlp()
bigrams, trigrams, dictionary = get_gensim_phrases_and_dictionary()
lda_model = get_lda_model()

tab1, tab2 = st.tabs(['Popularity Prediction', 'Visualizations'])

with tab1:
    with st.form('articleForm', clear_on_submit=False):

        titleText = st.text_input('Article Title', 'Input title/headline.')
        deckText = st.text_input('Article Subheading', 'Input subheading/drop head/deck that summarizes the article.')
        articleText = st.text_area('Article Text', 'Input the body of the news article.')
        hour = st.selectbox('Section/Category', sorted(_sections))

        st.markdown("""---""")

        dateInput = st.date_input('Date')

        timecol1, timecol2, timecol3 = st.columns(3)
        with timecol1:
            hour = st.selectbox('Hour', (f'{h:02}' for h in range(1, 12 + 1)))
        with timecol2:
            minute = st.selectbox('Minute', (f'{h:02}' for h in range(0, 60)))
        with timecol3:
            am_or_pm = st.selectbox('AM/PM', ['AM', 'PM'])

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
        # st.write(lda_model.print_topic(max(enumerate(topic_vector), key = lambda item : item[1])[0]))
        for relevant_topic, topic_weight in sorted(enumerate(topic_vector), key = lambda item : -item[1])[:3]:
            st.write(topic_weight, lda_model.print_topic(relevant_topic))