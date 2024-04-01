from typing import Optional, Type

from langchain.callbacks.manager import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from template.services.reddit_api_wrapper import RedditAPIWrapper
from template.tools.base import BaseTool


class RedditSearchSchema(BaseModel):
    query: str = Field(
        ...,
        description="The search query for Reddit search.",
    )


class RedditSearchTool(BaseTool):
    """Tool for the Reddit API."""

    name = "Reddit Search"

    slug = "reddit-search"

    description = "A wrapper around Reddit." "Useful for searching Reddit for posts."

    args_schema: Type[RedditSearchSchema] = RedditSearchSchema

    tool_id = "043489f8-ef05-4151-8849-7f954e4910be"

    def _run():
        pass

    async def _arun(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Search Reddit and return the results."""
        reddit = RedditAPIWrapper()
        return await reddit.search(query, subreddit="all")

    async def send_event(self, send, response_streamer, data):
        pass
