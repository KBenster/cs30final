import praw

#check all titles for stock tickers, get overall sentiment from comments and text.

reddit = praw.Reddit(
    client_id="my client id",
    client_secret="my client secret",
    user_agent="Mozilla/5.0 (Nintendo WiiU) AppleWebKit/536.30 (KHTML, like Gecko) NX/3.0.4.2.12 NintendoBrowser/4.3.1.11264.US"
)


def get_submissions(amount, page): #page can be controversial gilded hot new rising top
    submissions = {}
    match page:
        case "hot":
            for submission in reddit.subreddit("wallstreetbets").hot(limit=amount):
                submission_text = submission.title + ". " #submission comments and text included
                submission_link = submission.permalink
                submissions[submission_link] = submission_text
        case _:
            return submissions
    return submissions
