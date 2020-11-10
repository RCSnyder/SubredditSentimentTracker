from psaw import PushshiftAPI
import datetime as dt
import praw

TEST_START_DATE = int(dt.datetime(2020, 1, 1).timestamp())
TEST_END_DATE = int(dt.datetime(2020, 1, 2).timestamp())
TEST_MAX = 1
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
    # gets a test submission
    threads = list(get_submissions(TEST_SUBREDDIT, TEST_START_DATE, TEST_END_DATE, TEST_MAX))
    submission_id = threads[0].d_['id']
    thread_full_link = threads[0].d_['full_link']

    submission = get_comments_from_submission(submission_id)
    print(thread_full_link)
    for top_level_comment in submission.comments:
        print(top_level_comment.body)


test_get_comments()
test_get_submissions()
test_get_comments_from_submission()
