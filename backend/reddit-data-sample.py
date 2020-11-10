from psaw import PushshiftAPI
import datetime as dt
import praw

TEST_START_DATE = int(dt.datetime(2020, 2, 1).timestamp())
TEST_END_DATE = int(dt.datetime(2020, 2, 2).timestamp())
TEST_MAX = 100
MIN_COMMENTS = 10
TEST_SUBREDDIT = 'askreddit'

# PRAW OAuth stuff
CLIENT_ID = 'kdM81oo03fgHdQ'
CLIENT_SECRET = 'MSsQk9IBTimxRRMA0AZd2hxIpIS__w'
PASSWORD = 'Nolimit08212013'
USERAGENT = 'sentiment analysis script by /u/DentonPokerEnthusist'
USERNAME = 'DentonPokerEnthusist'

REDDIT = praw.Reddit(client_id=CLIENT_ID, client_secret=CLIENT_SECRET,
                     password=PASSWORD, user_agent=USERAGENT,
                     username=USERNAME)


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


def get_submissions(subreddit, start_date, end_date, limit):
    """returns a generator object of threads from a subreddit between a certain period of time"""
    api = PushshiftAPI()
    return api.search_submissions(after=start_date, before=end_date,
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

    api = PushshiftAPI()
    threads = list(api.search_submissions(after=start_date, before=end_date,
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



# test_get_comments()
# test_get_submissions()
# test_get_comments_from_submission()
# test_get_list_of_submission_dictionaries()
test_filter_list_of_dictionary_submission()
