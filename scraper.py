import praw
#Mozilla/5.0 (Nintendo WiiU) AppleWebKit/536.30 (KHTML, like Gecko) NX/3.0.4.2.12 NintendoBrowser/4.3.1.11264.US

#check all titles for stock tickers, get overall sentiment from comments and text.

reddit = praw.Reddit(
    client_id="my client id",
    client_secret="my client secret",
    user_agent="my user agent",
)