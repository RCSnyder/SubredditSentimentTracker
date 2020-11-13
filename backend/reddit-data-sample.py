import os

from psaw import PushshiftAPI
import datetime as dt
import praw
from praw.models import MoreComments
import time
import numpy as np
import random
import csv
import pandas as pd
import codecs
import re
import sys
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import json
from ibm_watson import ToneAnalyzerV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import ApiException
import flair
from segtok.segmenter import split_single

flair_sentiment = flair.models.TextClassifier.load('en-sentiment')
# nltk.download('vader_lexicon')

TEST_START_DATE = int(dt.datetime(2020, 11, 1, 0, 0).timestamp())
TEST_END_DATE = int(dt.datetime(2020, 11, 2, 0, 0).timestamp())
# print(TEST_END_DATE - TEST_START_DATE)
TEST_MAX = 100
MIN_COMMENTS = 500
TEST_SUBREDDIT = 'politics'

# PRAW OAuth stuff
CLIENT_ID = 'kdM81oo03fgHdQ'
CLIENT_SECRET = 'MSsQk9IBTimxRRMA0AZd2hxIpIS__w'
PASSWORD = 'Nolimit08212013'
USERAGENT = 'sentiment analysis script by /u/DentonPokerEnthusist'
USERNAME = 'DentonPokerEnthusist'

random.seed(hash('setting random seeds') % 2 ** 32 - 1)
np.random.seed(hash('improves reproducibility') % 2 ** 32 - 1)

REDDIT = praw.Reddit(client_id=CLIENT_ID, client_secret=CLIENT_SECRET,
                     password=PASSWORD, user_agent=USERAGENT,
                     username=USERNAME)

API = PushshiftAPI()

IBM_API_KEY = "OfhB70w2GIjB99k6zAF_k-ezpWK1evbmG0zk9Mfs35_v"
IBM_API = "ch7dMr1nkRWOKvNo_fw4exPB5CdeOAqvMsCjxxROa4up"
IBM_URL = "https://api.us-south.tone-analyzer.watson.cloud.ibm.com/instances/259eb6a9-2ccd-4a2b-aa91-6a233298d4ea"

authenticator = IAMAuthenticator(IBM_API)
TONE_ANALYZER = ToneAnalyzerV3(
    version='2017-09-21',
    authenticator=authenticator
)
TONE_ANALYZER.set_service_url(IBM_URL)


def get_historical_submissions(subreddit, limit):
    """returns a list of submission dictionaries from the past 30 months,
        querying a random 4 hour chunk in a random day of each month"""
    past_30_months = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
                      12, 11, 10, 9, 8, 7, 6, 5, 4, 3,
                      2, 1, 12, 11, 10, 9, 8, 7, 6, 5]
    all_submissions = []
    day = 0
    year = 2020
    hacky_year_flag = 0

    for month in past_30_months:
        # derive year
        if hacky_year_flag < 9:
            year = 2020
        if 9 < hacky_year_flag <= 21:
            year = 2019
        if hacky_year_flag > 22:
            year = 2018

        hacky_year_flag += 1

        # generate random day
        if month in [1, 3, 5, 7, 8, 10, 12]:
            day = random.randint(1, 31)
        if month in [4, 6, 9, 11]:
            day = random.randint(1, 30)
        if month in [2]:
            day = random.randint(1, 28)

        # generate random 4 hour time chunk
        start_hour = random.randint(0, 19)
        end_hour = start_hour + 4

        start_time = int(dt.datetime(year, month, day, start_hour, 0).timestamp())
        end_time = int(dt.datetime(year, month, day, end_hour, 0).timestamp())

        # gets submissions and adds submission dictionary to master list
        threads = list(get_submissions(subreddit, start_time, end_time, limit))
        for item in threads:
            all_submissions.append(item.d_)

        print('querying month:', hacky_year_flag)
        print('total submissions:', len(all_submissions))

    return all_submissions


def test_get_historical_submissions():
    submission_dictionary = get_historical_submissions(TEST_SUBREDDIT, TEST_MAX)

    num_comments = 0
    for submission in submission_dictionary:
        num_comments += submission['num_comments']

    print('total submissions:', len(submission_dictionary))
    print("total comments:", num_comments)


def save_historical_submission_comments(list_of_dictionary_submissions, file_name):
    """saves all of the comments from a list of dictionary submissions into a single column csv"""
    all_comments_list = []
    submission_count = 1

    for submission_dict in list_of_dictionary_submissions:
        print('saving comments from submission', submission_count, '/', len(list_of_dictionary_submissions))
        submission_count += 1
        submission = (REDDIT.submission(id=submission_dict['id']))

        submission.comments.replace_more(limit=None)
        for comment in submission.comments.list():
            temp_dict = {'body': comment.body, 'comment_id': comment}
            all_comments_list.append(temp_dict)
        print('total comments: ', len(all_comments_list))

    comments_df = pd.DataFrame(all_comments_list, columns=['body', 'comment_id'])

    print(comments_df)

    print('saving comments to file:', file_name, '...')
    comments_df.to_csv(file_name)
    print('done.')


def test_save_historical_submission_comments():
    """tests the save function with 3 submission comments"""
    data = []
    threads = list(get_submissions(TEST_SUBREDDIT, TEST_START_DATE, TEST_END_DATE, TEST_MAX))
    for item in threads:
        data.append(item.d_)

    save_historical_submission_comments(data, TEST_SUBREDDIT + '_TEST.csv')


def run_save_historical_data():
    """runs the full 30 month submission/comment scrape and saves it to a csv"""
    data = get_historical_submissions(TEST_SUBREDDIT, TEST_MAX)
    save_historical_submission_comments(data, TEST_SUBREDDIT + '_past_30_months_comments.csv')


def get_all_submissions_in_24_hours(subreddit, start_date, end_date, limit):
    """returns list of all submissions within 6, 4 hour increments in a twenty four hour period over MIN_COMMENTS."""
    list_of_all_submissions = []
    inc_start_date = start_date
    inc_end_date = end_date + 600

    for i in range(0, 86400, 14400):
        inc_end_date += i
        inc_start_date += i
        threads = list(get_submissions(subreddit, inc_start_date, inc_end_date, limit))
        for item in threads:
            if item.d_['num_comments'] > MIN_COMMENTS:
                list_of_all_submissions.append(item.d_)
        print(len(list_of_all_submissions))
    return list_of_all_submissions


def test_get_all_submissions_in_24_hours():
    """tests get_all_submissions_in_24_hours by printing out the length of submissions"""
    all_submissions = get_all_submissions_in_24_hours(TEST_SUBREDDIT, TEST_START_DATE, TEST_END_DATE, TEST_MAX)
    print(len(all_submissions))
    for x in all_submissions[0:3]:
        print(x)


def get_comments(subreddit, start_date, end_date, limit):
    """returns a generator object of comments from a subreddit between a certain period of time"""
    api = PushshiftAPI()
    return api.search_comments(after=start_date, before=end_date,
                               subreddit=subreddit, limit=limit
                               # , filter=['author', 'body', 'created_utc', 'nest_level']
                               )


def test_get_comments():
    """tests the get_comments function to return 1 comment between TEST_START_DATE and TEST_END_DATE"""
    comments = list(get_comments(TEST_SUBREDDIT, TEST_START_DATE, TEST_END_DATE, TEST_MAX))

    # prints the dictionary of variables for each comment
    for x in comments:
        print(x.d_)


def get_number_of_submissions():
    """prints out number of submissions in a subreddit in a given time period"""

    start = time.time()
    print("counting submissions in", TEST_SUBREDDIT, 'between', TEST_START_DATE, 'and', TEST_END_DATE)
    threads = list(get_submissions(TEST_SUBREDDIT, TEST_START_DATE, TEST_END_DATE, TEST_MAX))
    end = time.time()
    print('time elapsed: ', end - start)
    print('total submissions:', len(threads))
    print(TEST_MAX)


def get_submissions(subreddit, start_date, end_date, limit):
    """returns a generator object of threads from a subreddit between a certain period of time"""

    return API.search_submissions(after=start_date, before=end_date,
                                  subreddit=subreddit, limit=limit)


def test_get_submissions():
    """tests get_submissions function to return 1 thread between TEST_START_DATE and TEST_END_DATE"""
    threads = list(get_submissions(TEST_SUBREDDIT, TEST_START_DATE, TEST_END_DATE, TEST_MAX))

    # prints the dictionary of variables for each submission
    for x in threads:
        print(x.d_)


def get_comments_from_submission(submission_id):
    """returns a submission from a given submission id"""
    submission = (REDDIT.submission(id=submission_id))
    return submission


def test_get_comments_from_submission():
    """tests get_comments_from_submission by printing out the comments of a submission"""
    # gets a test submission
    threads = list(get_submissions(TEST_SUBREDDIT, TEST_START_DATE, TEST_END_DATE, TEST_MAX))
    submission_id = threads[0].d_['id']

    # prints link to thread
    thread_full_link = threads[0].d_['full_link']
    print(thread_full_link)

    # prints submission title
    thread_title = threads[0].d_['title']
    print(thread_title)

    submission = get_comments_from_submission(submission_id)
    for top_level_comment in submission.comments:
        print(top_level_comment.body)


def get_list_of_submission_dictionaries(subreddit, start_date, end_date, limit):
    """returns a list of dictionaries of each submission in a given subreddit between a period of time"""
    list_of__submission_dictionaries = []

    threads = list(API.search_submissions(after=start_date, before=end_date,
                                          subreddit=subreddit, limit=limit))

    # appends thread submission dictionary to a list
    for thread_submission in threads:
        list_of__submission_dictionaries.append(thread_submission.d_)

    return list_of__submission_dictionaries


def test_get_list_of_submission_dictionaries():
    submission_list = get_list_of_submission_dictionaries(TEST_SUBREDDIT, TEST_START_DATE, TEST_END_DATE, TEST_MAX)
    print(submission_list[0])
    print(len(submission_list))


def filter_list_of_dictionary_submission(submission_list, min_comments):
    """filters the list of submission dictionaries to only include submissions with more than min_comments comments"""
    filtered_submission_list = []
    # filter submission_list for submissions with > min_comments # comments
    for submission_dictionary in submission_list:
        if submission_dictionary['num_comments'] >= min_comments:
            filtered_submission_list.append(submission_dictionary)

    return filtered_submission_list


def test_filter_list_of_dictionary_submission():
    """prints length of the submission list before and after filtering by min_comments number"""
    submission_list = get_list_of_submission_dictionaries(TEST_SUBREDDIT, TEST_START_DATE, TEST_END_DATE, TEST_MAX)
    print(len(submission_list))

    filtered_list = filter_list_of_dictionary_submission(submission_list, MIN_COMMENTS)
    print(len(filtered_list))


def get_comments_from_submission_id(submission_id):
    """returns a list of all comment ids in a submission by submission id"""
    flat_comments = []
    tree_comments = []

    submission = (REDDIT.submission(id=submission_id))
    print(submission.num_comments)
    print(submission.shortlink)

    # sort comments by best and get the flattened list
    submission.comment_sort = 'confidence'

    # tree comments traversal
    submission.comments.replace_more(limit=1)
    for comm in submission.comments.list():
        tree_comments.append(comm)

    flat_comments = list(submission.comments)

    return flat_comments, tree_comments


def test_print_comments():
    """prints first 5 comments returned by get_comments_from_submission_id"""
    flat_comments, tree_comments = get_comments_from_submission_id('jrjn70')
    print(len(flat_comments))
    print(len(tree_comments))

    print('flat comments')
    for c in flat_comments[0:5]:
        comment_instance = REDDIT.comment(c)
        print(comment_instance.body)

    print()
    print('tree comments')
    for c in tree_comments[0:5]:
        comment_instance = REDDIT.comment(c)
        print(comment_instance.body)


def get_comments_by_percentage(submission_id, percent_of_comments):
    """returns a list of comment id that is a percentage of the total number of comments in a submission """
    comments_list = []
    submission = (REDDIT.submission(id=submission_id))
    max_comments = int(submission.num_comments * percent_of_comments)

    print(submission.num_comments)
    print(max_comments)

    comment_count = 0

    # sort comments by best and get list of id's
    submission.comment_sort = 'confidence'
    submission.comments.replace_more(limit=40)
    for comment_id in submission.comments.list():
        if comment_count >= max_comments:
            break
        comments_list.append(comment_id)
        comment_count += 1

    return comments_list


def test_get_comments_by_percentage():
    """tests get_comments_by_percentage"""
    comment_ids = get_comments_by_percentage('jrjn70', .10)

    print(len(comment_ids))

    for c in comment_ids[0:5]:
        comment_instance = REDDIT.comment(c)
        print(comment_instance.body)


def read_csv_to_dataframe(file_name):
    """reads a csv file into a dataframe and drops the redundant index column"""
    df = pd.read_csv(file_name)
    df = df.drop(['Unnamed: 0'], axis=1)
    return df


def test_read_csv_to_dataframe(fname):
    """tests the read csv file function"""
    df = read_csv_to_dataframe(fname)
    print(df.head())


def sanitize_characters(raw_input_file, clean_output_file):
    """given a csv file removes errors in ascii encoding, drops the errors and writes to a clean file"""
    input_file = codecs.open(raw_input_file, 'r', encoding='ascii', errors='ignore')
    output_file = open(clean_output_file, 'w', encoding='ascii', errors='ignore')

    for line in input_file:
        # removes extra newline
        line = line.rstrip('\n')
        output_file.write(line)


def run_sanitize_characters():
    """runs clean_comments() with the politics_past_30_months_comments.csv and prints head of cleaned file"""
    sanitize_characters('politics_past_30_months_comments.csv', 'politics_past_30_months_comments_cleaned.csv')

    df = pd.read_csv('politics_past_30_months_comments_cleaned.csv')
    df = df.drop(['Unnamed: 0'], axis=1)
    print(df.head())


def standardize_comments(df, column_name):
    """standardizes the comment bodies for sentiment analysis and tone analysis can be performed """
    # create a copy of the dataframe
    df_copy = df.copy()

    # remove rows that contain '[deleted]' or '[removed]' in the comment body
    df_copy = df_copy[(df[column_name] != '[removed]') & (df[column_name] != '[deleted]')]

    # remove rows with null values
    df_copy.dropna(inplace=True)

    # remove rows with the bot comment that starts with "Register to vote"
    df_copy = df_copy[~df_copy[column_name].str.startswith('Register to vote')]

    # remove rows with 'Thank you for participating in /r/Politics' in the body
    df_copy = df_copy[~df_copy[column_name].str.contains('Thank you for participating in /r/Politics')]

    # remove rows that contain 'I am a bot' in the comment body
    df_copy = df_copy[~df_copy[column_name].str.contains('I am a bot')]

    # replace characters in comment bodies
    df_copy[column_name] = df_copy[column_name].str.replace(r"http\S+", "")
    df_copy[column_name] = df_copy[column_name].str.replace(r"http", "")
    df_copy[column_name] = df_copy[column_name].str.replace(r"@\S+", "")
    df_copy[column_name] = df_copy[column_name].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df_copy[column_name] = df_copy[column_name].str.replace(">", "")
    df_copy[column_name] = df_copy[column_name].str.replace("    ", " ")
    df_copy[column_name] = df_copy[column_name].str.replace("   ", " ")
    df_copy[column_name] = df_copy[column_name].str.replace("  ", " ")
    df_copy[column_name] = df_copy[column_name].str.replace("\"", "")
    df_copy[column_name] = df_copy[column_name].str.replace(r"@", "at")
    df_copy[column_name] = df_copy[column_name].str.lower()

    # remove rows with empty comment strings
    df_copy = df_copy[df_copy[column_name] != '']

    # remove rows with comment strings containing only a space
    df_copy = df_copy[df_copy[column_name] != ' ']

    # needs to figure out a way to get rid of this pattern of linking in comment bodies
    #  [your submission](https://redd.it/jdn4dx)
    #  [Dripping with money: A look behind the gated affluence of Trumps Palm Beach](https://redd.it/jdn4dx)
    #  [No Queue Flooding](https://www.reddit.com/r/politics/wiki/index#wiki_do_not_flood_the_new_queue.)

    return df_copy


def run_standardize_comments():
    """runs standardize comments function and saves it into a new file"""
    df = pd.read_csv('politics_past_30_months_comments_cleaned.csv')
    df = df.drop(['Unnamed: 0'], axis=1)

    standardized_df = standardize_comments(df, 'body')
    print(standardized_df.head())
    print()
    print('original length:', len(df))
    print('standardized length:', len(standardized_df))
    print('removed', len(df) - len(standardized_df), 'comments')

    # THIS MIGHT BRING BACK THE UTF-8 ENCODING EMOJIS. MIGHT HAVE TO WRITE TO CSV IN ASCII
    standardized_df.to_csv('politics_past_30_months_comments_cleaned_standardized.csv')


def add_vader_sentiment_scores(df):
    """given a dataframe with 'body' as the column name for comment bodies
       calculates the nltk vader sentiment scores and adds them to
       'vader_compound_score', 'vader_negative_score', 'vader_neutral_score', 'vader_positive_score'"""
    sid = SentimentIntensityAnalyzer()
    df_copy = df.copy()

    df_copy['vader_compound_score'] = [sid.polarity_scores(str(x))['compound'] for x in df_copy['body']]
    df_copy['vader_negative_score'] = [sid.polarity_scores(str(x))['neg'] for x in df_copy['body']]
    df_copy['vader_neutral_score'] = [sid.polarity_scores(str(x))['neu'] for x in df_copy['body']]
    df_copy['vader_positive_score'] = [sid.polarity_scores(str(x))['pos'] for x in df_copy['body']]

    return df_copy


def run_add_vader_sentiment_scores():
    """runs add_vader_sentiment_scores"""
    df = pd.read_csv('politics_past_30_months_comments_cleaned_standardized.csv', encoding='utf-8')
    df = df.drop(['Unnamed: 0'], axis=1)
    scores_df = add_vader_sentiment_scores(df)
    scores_df.to_csv('politics_past_30_months_comments_cleaned_standardized_vader.csv')


def get_tone_from_IBM(comment):
    """sends an api request to IBM to get back a json object"""
    try:
        tone_analysis = TONE_ANALYZER.tone(
            {'text': comment},
            content_type='application/json'
        ).get_result()
        # print(json.dumps(tone_analysis, indent=2))
        return tone_analysis
    except ApiException as ex:
        print("Method failed with status code " + str(ex.code) + ": " + ex.message)


def get_columns_from_IBM_tone(tone_info_dictionary):
    """parses the dictionary returned by IBM to return the values of the overall document tone in order
        'Anger', 'Fear', 'Joy', 'Sadness', 'Analytical', 'Confident', 'Tentative'
        if no value assigned by IBM will return 'None'"""

    anger = 0
    fear = 0
    joy = 0
    sadness = 0
    analytical = 0
    confident = 0
    tentative = 0

    for tone in tone_info_dictionary['document_tone']['tones']:
        if tone['tone_id'] == 'anger':
            anger = tone['score']
        if tone['tone_id'] == 'fear':
            fear = tone['score']
        if tone['tone_id'] == 'joy':
            joy = tone['score']
        if tone['tone_id'] == 'sadness':
            sadness = tone['score']
        if tone['tone_id'] == 'analytical':
            analytical = tone['score']
        if tone['tone_id'] == 'confident':
            confident = tone['score']
        if tone['tone_id'] == 'tentative':
            tentative = tone['score']

    return [anger, fear, joy, sadness, analytical, confident, tentative]


def test_get_tone_from_IBM():
    """runs get_tone_from_IBM and prints out response object"""
    comments = ["This was a really sucky movie. I will probably never go see this movie ever again. I am going to "
                "tell my whole family never to watch this movie. I very much enjoyed the special cameo in it "
                "though. I loved the plot line."]
    tone_info_dictionary = get_tone_from_IBM(comments[0])

    tones = get_columns_from_IBM_tone(tone_info_dictionary)
    print(tones)


def get_tone_from_api_and_return_columns(comment):
    """given a comment, calls the IBM tone api and returns the column values"""
    tone_dict = get_tone_from_IBM(comment)
    return get_columns_from_IBM_tone(tone_dict)


def test_get_tone_from_api_and_return_columns():
    comments = ["oh look, the soft power it's gone all soft maybe donald can get those fruit drops back from angela, "
                "that'll fix things and from boris and emmanuel and scott and maybe buy some aluminium from jacinda "
                "and oh god, perhaps an apology to justin and come to think of it there might be a problem",
                "now we need a few republicans to indicate they don't stand with fascism anyone?",
                "nancy pelosi the president has to realize that the words of the president of the united states weigh "
                "a ton, pelosi told abc news and, in our political dialogue, to inject fear tactics into it, "
                "especially a woman governor and her family, is so irresponsible and, in all fairness to people who "
                "listen to him, people think the president is important and what he says should be adhered to chris "
                "coons particularly troubling, tome perez fanning the flames of division, and slamming him for "
                "lacking a plan on covid 19 and the economy if i were roundly condemning trump, i go quite a bit "
                "farther than irresponsible and troubling accuse him of the obvious criminality, ffs it's more than "
                "irresponsibility, he's doing it purposefully these frogs have been boiled i guess ",
                "used the same chant for hillary because his base can't think past 3 syllables how many of his chants "
                "are 3 syllables?",
                "hilariously, texas is one of a number of states that restricts political clothing near polling "
                "places so in this county poll workers will be required to turn away someone wearing a joe biden "
                "shirt or a maga hat, but will be fined for turning away someone without a mask makes total sense ",
                "how about looking at this from the other side unmasked voters should be charged with felony voter "
                "intimidation ", "ironic, coming from the poster child state for voter suppression "]
    for x in comments:
        print(get_tone_from_api_and_return_columns(x))


def add_tone_columns_to_csv(input_file_name, output_file_name):
    """given a csv file, runs ibm's tone analysis on the 'body' column and saves the outputs to the columns,
    'Anger', 'Fear', 'Joy', 'Sadness', 'Analytical', 'Confident', 'Tentative'"""
    df = pd.read_csv(input_file_name)
    df = df.drop(['Unnamed: 0'], axis=1)

    # df['anger'], df['fear'], df['joy'], df['sadness'], df['analytical'], df['confident'], df['tentative']
    df[['anger', 'fear', 'joy', 'sadness', 'analytical', 'confident', 'tentative']] = \
        pd.DataFrame(df['body'].apply(get_tone_from_api_and_return_columns).tolist())

    df.to_csv(output_file_name)


def run_add_tone_columns_to_csv():
    """tests the add_tone_columns_to_csv() function with
    'test_data_for_tone.csv'
    'test_data_for_tone_added.csv'"""
    add_tone_columns_to_csv('test_data_for_tone.csv', 'test_data_for_tone_added.csv')


def get_whole_flair_sentiment(comment):
    """given a comment body, gets the sentiment score on the entire comment.
       returns the whole_comment_sentiment score"""
    text = flair.data.Sentence(comment)
    flair_sentiment.predict(text)
    value = text.labels[0].to_dict()['value']
    if value == 'POSITIVE':
        whole_comment_sentiment = text.to_dict()['labels'][0]['confidence']
    else:
        whole_comment_sentiment = -(text.to_dict()['labels'][0]['confidence'])

    whole_comment_sentiment = round(whole_comment_sentiment, 6)

    return whole_comment_sentiment


def test_get_whole_flair_sentiment():
    """runs get_flair_sentiment with a test comment list and prints the results and results_sum"""

    comments = ["This was a really sucky movie. I will probably never go see this movie ever again. I am going to "
                "tell my whole family never to watch this movie. I very much enjoyed the special cameo in it "
                "though. I loved the plot line."]
    for x in comments:
        result_sum = get_whole_flair_sentiment(x)
        print(x)
        print('Whole comment sentiment:', result_sum)
        print()


def make_sentences(comment):
    """ Break apart text into a list of sentences """
    sentences = [sent for sent in split_single(comment)]
    return sentences


def test_make_sentences():
    """tests make_sentences with a 2 sentence comment"""
    long_comment = ['I think this movie was really good and will go and see it again. '
                    'This movie really sucked and I hated it']
    new_sentences = make_sentences(long_comment[0])
    print(new_sentences)


def get_sentence_sentiments(comment):
    """given a comment, splits the comment into sentences and returns the list of sentiment scores"""
    sentence_score_list = []

    split_comment = make_sentences(comment)
    for sentence in split_comment:
        if sentence == ' ' or sentence == '' or sentence == '  ':
            continue
        text = flair.data.Sentence(sentence)
        flair_sentiment.predict(text)

        value = text.labels[0].to_dict()['value']
        if value == 'POSITIVE':
            result = text.to_dict()['labels'][0]['confidence']
        else:
            result = -(text.to_dict()['labels'][0]['confidence'])

        sentence_score = round(result, 6)
        sentence_score_list.append(sentence_score)

    return sentence_score_list


def test_get_sentence_sentiments():
    """tests the get_sentence_sentiments() function"""
    long_comment = ["This was a really sucky movie. I will probably never go see this movie ever again. I am going to "
                    "tell my whole family never to watch this movie. I very much enjoyed the special cameo in it "
                    "though. I loved the plot line."]

    sentence_score_list = get_sentence_sentiments(long_comment[0])
    print(long_comment[0])
    print('per sentence sentiment:', sentence_score_list)
    print()


def get_whole_and_per_sentence_flair_sentiments(list_of_comments):
    """given a list of variable length comments, gets the whole comment sentiment and the per sentence sentiment"""

    for comment in list_of_comments:
        result_sum = get_whole_flair_sentiment(comment)
        print(comment)
        print('Whole comment sentiment:', result_sum)
        print()
        sentence_score_list = get_sentence_sentiments(comment)
        print(comment)
        print('per sentence sentiment:', sentence_score_list)
        print()


def test_get_whole_and_per_sentence_flair_sentiments():
    """tests getting the whole comment sentiment and the list of each sentence sentiment
    given a list of comments of variable length"""
    long_comments = ["This was a really sucky movie. I will probably never go see this movie ever again. I am going to "
                     "tell my whole family never to watch this movie. I very much enjoyed the special cameo in it "
                     "though. I loved the plot line.",

                     "it's intended to make the polling places dangerous by contaminating the air inside with virus "
                     "that can linger for hours",

                     "simple, just create an unmasked line in a separate part of the location let them infect each "
                     "other"]
    get_whole_and_per_sentence_flair_sentiments(long_comments)


def add_flair_sentiment_to_csv(input_file_name, output_file_name):
    """given a file_name.csv of comments with the column name 'body',
    calculates the whole comment sentiment score and the per sentence sentiment score
    and appends it to a new filename.csv
    VERY SLOW"""

    df = pd.read_csv(input_file_name)
    df = df.drop(['Unnamed: 0'], axis=1)

    df['whole_comment_sentiment_flair'] = df['body'].apply(get_whole_flair_sentiment)
    df['per_sentence_sentiment_flair'] = df['body'].apply(get_sentence_sentiments)

    df.to_csv(output_file_name)


def run_add_flair_sentiment_to_csv():
    """runs the add_flair_sentiment_to_csv() with politics_past_30_months_comments_cleaned_standardized_vader.csv
    and politics_past_30_months_comments_cleaned_standardized_vader_flair.csv"""
    add_flair_sentiment_to_csv('politics_past_30_months_comments_cleaned_standardized_vader.csv',
                               'politics_past_30_months_comments_cleaned_standardized_vader_flair.csv')


def get_comment_information_by_id(comment_id):
    """prints out whats available to a reddit comment by comment id"""
    comment = REDDIT.comment(comment_id)
    print(comment.body)
    print(vars(comment))


def test_get_comment_information_by_id():
    """tests get_comment_information_by_id() with a sample comment id"""
    get_comment_information_by_id('g99c7c0')


def get_specific_comment_info(comment_id):
    """Given a comment id, read in the comment using praw and extracts,
    created_utc
    permalink
    score
    link_id"""
    start = time.time()

    comment = REDDIT.comment(comment_id)

    end = time.time()
    print(end - start)
    return comment.created_utc, comment.permalink, comment.score, comment.link_id


def test_get_specific_comment_info():
    """tests get_specific_comment_info() with a sample reddit comment id"""
    a, b, c, d = get_specific_comment_info('g99c7c0')
    print('time created:', a, 'type:', type(a))
    print('permalink:', b, 'type:', type(b))
    print('karma score:', c, 'type:', type(c))
    print('submission id:', d, 'type:', type(d))


def add_time_created_permalink_karma_submission_id(input_file_name, output_file_name):
    """reads in a csv file and based on 'comment_id' value, adds the return values from
     get_specific_comment_info() to new columns"""

    df = pd.read_csv(input_file_name)
    df = df.drop(['Unnamed: 0'], axis=1)

    df['created_utc'], df['permalink'], df['score'], df['link_id'] = df['comment_id'].apply(get_specific_comment_info)

    df.to_csv(output_file_name)


def run_add_time_created_permalink_karma_submission_id():
    """runs the add_time_created_permalink_karma_submission_id() function with
    politics_past_30_months_comments_cleaned_standardized_vader_flair.csv
    politics_past_30_months_comments_cleaned_standardized_vader_flair_info.csv"""
    add_time_created_permalink_karma_submission_id('politics_past_30_months_comments_cleaned_standardized_vader_flair'
                                                   '.csv',
                                                   'politics_past_30_months_comments_cleaned_standardized_vader_flair'
                                                   '_info.csv')


# test_get_comments()
# test_get_submissions()
# test_get_comments_from_submission()
# test_get_list_of_submission_dictionaries()
# test_filter_list_of_dictionary_submission()
# test_print_comments()
# test_get_comments_by_percentage()
# get_number_of_submissions()
# test_get_all_submissions_in_24_hours()
# test_get_historical_submissions()
# test_save_historical_submission_comments()
# run_save_historical_data()
# test_read_csv_to_dataframe()
# run_sanitize_characters()
# run_standardize_comments()
# run_add_vader_sentiment_scores()
# test_get_tone_from_IBM()
# test_make_sentences()
# test_get_whole_flair_sentiment()
# test_get_sentence_sentiments()
# test_get_whole_and_per_sentence_flair_sentiments()
# run_add_flair_sentiment_to_csv()
# test_get_comment_information_by_id()
# test_get_specific_comment_info()
# run_add_time_created_permalink_karma_submission_id()
test_get_tone_from_api_and_return_columns()
run_add_tone_columns_to_csv()
