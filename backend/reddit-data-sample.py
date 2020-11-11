from psaw import PushshiftAPI
import datetime as dt
import praw
from praw.models import MoreComments
import time
import numpy as np
import random

TEST_START_DATE = int(dt.datetime(2019, 12, 1, 0, 0).timestamp())
TEST_END_DATE = int(dt.datetime(2020, 12, 1, 0, 0).timestamp())
print(TEST_END_DATE - TEST_START_DATE)
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


# test_get_comments()
# test_get_submissions()
# test_get_comments_from_submission()
# test_get_list_of_submission_dictionaries()
# test_filter_list_of_dictionary_submission()
# test_print_comments()
# test_get_comments_by_percentage()

# get_number_of_submissions()

# test_get_all_submissions_in_24_hours()

test_get_historical_submissions()
