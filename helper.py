import numpy as np
import pytz
import datetime

DAILY_SECONDS = 86400
HOURLY_SECONDS = 3600
N_TIME_GAUSSIANS = 4

def gaussian(x, center, width):
    return np.exp(-(x - center)**2 / (2 * width **2))

def periodic_gaussian(x, center_in_hours, width_in_hours):

    center = center_in_hours * HOURLY_SECONDS
    width = width_in_hours * HOURLY_SECONDS
    return gaussian(x - DAILY_SECONDS, center, width) + gaussian(x, center, width) + gaussian(x + DAILY_SECONDS, center, width)

def time_process(date, hour, minute, am_or_pm, timezone):

    time_dict = dict()

    tz = pytz.timezone(timezone)
    inputTime = tz.localize(datetime.datetime.combine(date, datetime.datetime.strptime(f'{hour}:{minute} {am_or_pm}', '%I:%M %p').time()))
    time_as_Eastern = inputTime.astimezone(pytz.timezone('US/Eastern'))

    dayofweek = time_as_Eastern.weekday()
    time_dict['dayofweek'] = dayofweek

    seconds_since_midnight = (time_as_Eastern - time_as_Eastern.replace(hour = 0, minute = 0, second = 0, microsecond = 0)).total_seconds()
    for basis_index in range(N_TIME_GAUSSIANS):
        hour = basis_index * 24 // N_TIME_GAUSSIANS
        time_dict[f'hour_{hour}'] = periodic_gaussian(seconds_since_midnight, hour, 24 // N_TIME_GAUSSIANS)

    return time_dict

def text_len(text):
    if text is None:
        return None
    return len(text.strip().split()) or None

def percentile_rank(score, population):
    return (population < score).sum() / len(population)

def number_to_ordinal(number: int) -> str:
    if number % 100 in [11, 12, 13]:
        return f'{number}th'
    suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(number % 10, 'th')
    return f'{number}{suffix}'

# books/review, your-money, and special-series are not encoded because they were not in the training data
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

_feature_columns = ['tweet_has_video',
                    'tweet_has_photo',
                    'article_has_video',
                    'article_has_audio',
                    'comments',
                    'tweetlength',
                    'titlelength',
                    'summarylength',
                    'articlelength',
                    'section',
                    'dayofweek',
                    'hour_0',
                    'hour_6',
                    'hour_12',
                    'hour_18',
                    'topic_000',
                    'topic_001',
                    'topic_002',
                    'topic_003',
                    'topic_004',
                    'topic_005',
                    'topic_006',
                    'topic_007',
                    'topic_008',
                    'topic_009',
                    'topic_010',
                    'topic_011',
                    'topic_012',
                    'topic_013',
                    'topic_014',
                    'topic_015',
                    'topic_016',
                    'topic_017',
                    'topic_018',
                    'topic_019',
                    'topic_020',
                    'topic_021',
                    'topic_022',
                    'topic_023',
                    'topic_024',
                    'topic_025',
                    'topic_026',
                    'topic_027',
                    'topic_028',
                    'topic_029',
                    'topic_030',
                    'topic_031',
                    'topic_032',
                    'topic_033',
                    'topic_034',
                    'topic_035',
                    'topic_036',
                    'topic_037',
                    'topic_038',
                    'topic_039',
                    'topic_040',
                    'topic_041',
                    'topic_042',
                    'topic_043',
                    'topic_044',
                    'topic_045',
                    'topic_046',
                    'topic_047',
                    'topic_048',
                    'topic_049',
                    'topic_050',
                    'topic_051',
                    'topic_052',
                    'topic_053',
                    'topic_054',
                    'topic_055',
                    'topic_056',
                    'topic_057',
                    'topic_058',
                    'topic_059',
                    'topic_060',
                    'topic_061',
                    'topic_062',
                    'topic_063',
                    'topic_064',
                    'topic_065',
                    'topic_066',
                    'topic_067',
                    'topic_068',
                    'topic_069',
                    'topic_070',
                    'topic_071',
                    'topic_072',
                    'topic_073',
                    'topic_074',
                    'topic_075',
                    'topic_076',
                    'topic_077',
                    'topic_078',
                    'topic_079',
                    'topic_080',
                    'topic_081',
                    'topic_082',
                    'topic_083',
                    'topic_084',
                    'topic_085',
                    'topic_086',
                    'topic_087',
                    'topic_088',
                    'topic_089',
                    'topic_090',
                    'topic_091',
                    'topic_092',
                    'topic_093',
                    'topic_094',
                    'topic_095',
                    'topic_096',
                    'topic_097',
                    'topic_098',
                    'topic_099',
                    'topic_100',
                    'topic_101',
                    'topic_102',
                    'topic_103',
                    'topic_104',
                    'topic_105',
                    'topic_106',
                    'topic_107',
                    'topic_108',
                    'topic_109',
                    'topic_110',
                    'topic_111',
                    'topic_112',
                    'topic_113',
                    'topic_114',
                    'topic_115',
                    'topic_116',
                    'topic_117',
                    'topic_118',
                    'topic_119',
                    'topic_120',
                    'topic_121',
                    'topic_122',
                    'topic_123',
                    'topic_124',
                    'topic_125',
                    'topic_126',
                    'topic_127',
                    'topic_128',
                    'topic_129'
                    ]

_topics = [feature for feature in _feature_columns if 'topic' in feature]

_select_categoricals = {
    'Article includes video?' : 'article_has_video',
    'Article includes audio?' : 'article_has_audio',
    'Enable reader comments?' : 'comments',
    'Tweet includes video?' : 'tweet_has_video',
    'Tweet includes photograph?' : 'tweet_has_photo',
    'Day of week' : 'dayofweek',
    'Time of day' : 'hour'
}

# Maybe namedtuple or dataclass would be more appropriate here
_conditional_proportions_options = {
    'tweet_has_video' : dict(
        xlabel = 'Does the tweet include a video?',
        order = [0, 1],
        xticks = ['No', 'Yes'],
        aspect = 1.25
    ),
    'tweet_has_photo' : dict(
        xlabel = 'Does the tweet include a photo?',
        order = [0, 1],
        xticks = ['No', 'Yes'],
        aspect = 1.25
    ),
    'article_has_video' : dict(
        xlabel = 'Does the article include a video?',
        order = [0, 1],
        xticks = ['No', 'Yes'],
        aspect = 1.25
    ),
    'article_has_audio' : dict(
        xlabel = 'Does the article include audio?',
        order = [0, 1],
        xticks = ['No', 'Yes'],
        aspect = 1.25
    ),
    'comments' : dict(
        xlabel = 'Are readers allowed to comment on the article webpage?',
        order = [0, 1],
        xticks = ['No', 'Yes'],
        aspect = 1.25
    ),
    'dayofweek' : dict(
        xlabel = 'Day of the week posted on Twitter',
        order = range(7),
        xticks = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        aspect = 3
    ),
    'hour' : dict(
        xlabel = 'Time of Twitter post',
        order = range(24),
        xticks = ['12 AM'] + [f'{h} AM' for h in range(1, 12)] + ['12 PM'] + [f'{h} PM' for h in range(1, 12)],
        aspect = 4
    )
}