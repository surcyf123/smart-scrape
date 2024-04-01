import asyncpraw
import os
import template

REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET")
REDDIT_USERNAME = os.environ.get("REDDIT_USERNAME")

if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET or not REDDIT_USERNAME:
    raise ValueError("Reddit client ID, client secret and username are required")


# User agent rule: https://github.com/reddit-archive/reddit/wiki/API#rules
REDDIT_USER_AGENT = (
    f"User-Agent: python:smart_scrape:v{template.__version__} (by /u/{REDDIT_USERNAME})"
)


class RedditAPIWrapper:
    async def search(
        self,
        query: str,
        subreddit: str = "all",
    ):
        async with asyncpraw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT,
        ) as reddit:
            submission = await reddit.subreddit(subreddit)

            posts = []

            async for post in submission.search(query, sort="new"):
                posts.append(
                    {
                        "subreddit": post.subreddit.display_name,
                        "title": post.title,
                        "url": post.url,
                    }
                )

            return posts
