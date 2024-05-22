from typing import List
from .reward import BaseRewardModel, BaseRewardEvent
from .config import RewardModelType
from neurons.validators.reward.reward_llm import RewardLLM
from datura.protocol import ScraperStreamingSynapse, ScraperTextRole
import traceback
import bittensor as bt
from neurons.validators.apify.web_scraper_actor import WebScraperActor
import re
import asyncio
from neurons.validators.utils.prompts import (
    SearchSummaryRelevancePrompt,
    extract_score_and_explanation,
)
import random
import json
from neurons.validators.utils.prompts import ScoringPrompt, SearchSummaryRelevancePrompt
import time

APIFY_LINK_SCRAPE_AMOUNT = 7


class WebSearchContentRelevanceModel(BaseRewardModel):
    reward_model_name: str = "VMware/open-llama-7b-open-instruct"

    @property
    def name(self) -> str:
        return RewardModelType.search_content_relevance.value

    def __init__(self, device: str, scoring_type: None, llm_reward: RewardLLM):
        super().__init__()
        self.device = device
        self.reward_llm = llm_reward

        self.scoring_type = scoring_type

    async def llm_process_validator_links(self, prompt, links_with_metadata):
        scoring_messages = []

        for link_with_metadata in links_with_metadata:
            url = link_with_metadata.get("url")
            title = link_with_metadata.get("title")
            description = link_with_metadata.get("description")
            content = f"Page Title: {title}. Page Description: {description}"

            result = self.get_scoring_text(
                prompt=prompt, content=content, response=None
            )
            if result:
                scoring_prompt, scoring_text = result
                scoring_messages.append({url: scoring_text})

        score_responses = self.reward_llm.llm_processing(scoring_messages)
        return score_responses

    async def process_links(
        self, prompt: str, responses: List[ScraperStreamingSynapse]
    ):
        all_links = []
        start_time = time.time()

        for response in responses:
            links = [
                link
                for link in random.sample(
                    response.search_completion_links,
                    min(
                        APIFY_LINK_SCRAPE_AMOUNT, len(response.search_completion_links)
                    ),
                )
            ]

            all_links.extend(links)

        unique_links = list(set(all_links))

        if len(unique_links) == 0:
            bt.logging.info("No unique links found to process.")
            return {}

        links_with_metadata = await WebScraperActor().scrape_metadata(urls=unique_links)

        for response in responses:
            for link_with_metadata in links_with_metadata:
                url = link_with_metadata.get("url")

                if url in response.search_completion_links:
                    response.validator_links.append(link_with_metadata)

        val_score_responses = await self.llm_process_validator_links(
            prompt, links_with_metadata
        )

        end_time = time.time()
        bt.logging.info(
            f"Fetched Web links method took {end_time - start_time} seconds. "
            f"All links count: {len(all_links)}, Unique links count: {len(unique_links)}, "
            f"APIFY fetched web links count: {len(links_with_metadata)}"
        )
        fetched_links = {link.get("url") for link in links_with_metadata}
        non_fetched_links = [link for link in unique_links if link not in fetched_links]

        bt.logging.info(
            f"Web links not fetched amount: {len(non_fetched_links)}; List: {non_fetched_links}; For prompt: [{prompt}]"
        )
        if len(non_fetched_links):
            bt.logging.info(
                f"Unique Web Links Amount: {len(unique_links)}; List: {unique_links};"
            )

        return val_score_responses

    def check_response_random_link(self, response: ScraperStreamingSynapse):
        try:
            completion = self.get_successful_search_summary_completion(
                response=response
            )

            if not completion:
                return 0

            search_completion_links = response.search_completion_links
            validator_links = response.validator_links

            if not search_completion_links or not validator_links:
                return 0

            if len(search_completion_links) < 2:
                # at least miners should provide two search links
                return 0

            # Google search results are separate because they include links with different domains from search
            google_search_results = str(response.search_results) + str(
                response.google_news_search_results
            )

            domain_to_search_result = {
                "arxiv.org": response.arxiv_search_results,
                "wikipedia.org": response.wikipedia_search_results,
                "reddit.com": response.reddit_search_results,
                "news.ycombinator.com": response.hacker_news_search_results,
                "youtube.com": response.youtube_search_results,
            }

            link_scores = []

            for val_link in validator_links:
                url = val_link.get("url")

                if not url:
                    link_scores.append(0)
                    continue

                domain_parts = url.split("/")[2].split(".")
                domain = ".".join(domain_parts[-2:])  # Extract the main domain

                if domain in domain_to_search_result:
                    if (
                        url in str(domain_to_search_result[domain])
                        or url in google_search_results
                    ):
                        link_scores.append(1)
                    else:
                        link_scores.append(0)
                else:
                    link_scores.append(1 if url in google_search_results else 0)

            if link_scores:
                return sum(link_scores) / len(link_scores)

            return 0
        except Exception as e:
            bt.logging.error(f"check_response_random_link: {str(e)}")
            return 0

    def get_scoring_text(
        self, prompt: str, content: str, response: ScraperStreamingSynapse
    ) -> BaseRewardEvent:
        try:
            if response:
                completion = self.get_successful_search_summary_completion(
                    response=response
                )

                if not completion:
                    return None

            if content is None:
                bt.logging.debug("Search Content is empty.")
                return None

            scoring_prompt_text = None
            scoring_prompt = SearchSummaryRelevancePrompt()

            if not scoring_prompt_text:
                scoring_prompt_text = scoring_prompt.text(prompt, content)

            return scoring_prompt, [
                {"role": "system", "content": scoring_prompt.get_system_message()},
                {"role": "user", "content": scoring_prompt_text},
            ]
        except Exception as e:
            bt.logging.error(f"Error in Prompt reward method: {str(e)}")
            return None

    def get_rewards(
        self, prompt: str, responses: List[ScraperStreamingSynapse], name: str, uids
    ) -> List[BaseRewardEvent]:
        try:
            completions: List[str] = self.get_successful_search_completions(responses)
            bt.logging.debug(
                f"WebSearchContentRelevanceModel | Calculating {len(completions)} rewards (typically < 1 sec/reward)."
            )
            bt.logging.trace(
                f"WebSearchContentRelevanceModel | prompt: {repr(prompt[:50])} ... {repr(prompt[-50:])}"
            )
            val_score_responses = asyncio.get_event_loop().run_until_complete(
                self.process_links(prompt=prompt, responses=responses)
            )

            bt.logging.info(
                f"WebSearchContentRelevanceModel | Keys in val_score_responses: {len(val_score_responses.keys()) if val_score_responses else 'No val_score_responses available'}"
            )
            scores = [
                self.check_response_random_link(response) for response in responses
            ]

            reward_events = []
            scoring_prompt = ScoringPrompt()

            grouped_val_score_responses = []

            for apify_score, response, uid_tensor in zip(scores, responses, uids):
                uid = uid_tensor.item()
                reward_event = BaseRewardEvent()
                reward_event.reward = 0

                response_scores = {}
                total_score = 0
                num_links = len(response.validator_links)
                max_links_considered = (
                    len(response.validator_links)
                    if len(response.validator_links) > 10
                    else 10
                )

                if num_links > 0:
                    for val_link in response.validator_links:
                        val_url = val_link.get("url")
                        if val_score_responses:
                            score_result = val_score_responses.get(val_url, None)
                            if score_result is not None:
                                score = scoring_prompt.extract_score(score_result)
                                total_score += (
                                    score / 10.0
                                )  # Adjust score scaling as needed
                                response_scores[val_url] = score

                    if total_score > 0:
                        average_score = total_score / max_links_considered
                        reward_event.reward = self.calculate_adjusted_score(
                            links_count=len(response.search_completion_links),
                            score=average_score,
                        )
                else:
                    bt.logging.info(f"UID '{uid}' has no validator links.")
                    reward_event.reward = 0  # Handle case with no validator links
                reward_event.reward = min(reward_event.reward * apify_score, 1)
                reward_events.append(reward_event)
                grouped_val_score_responses.append(response_scores)

                zero_scores = {}
                non_zero_scores = {}

            for (index, response), uid_tensor, reward_e in zip(
                enumerate(responses), uids, reward_events
            ):
                uid = uid_tensor.item()
                if reward_e.reward == 0:
                    # score_explain = score_responses.get(str(uid), "")
                    zero_scores[uid] = reward_e.reward
                else:
                    non_zero_scores[uid] = reward_e.reward

            bt.logging.info(
                f"==================================Web Search Content Relevance scoring Zero Scores  ({len(zero_scores)} cases)=================================="
            )
            bt.logging.info(json.dumps(zero_scores))
            bt.logging.info(
                f"==================================Web Search Content Relevance scoring Non-Zero Scores ({len(non_zero_scores)} cases)=================================="
            )
            bt.logging.info(json.dumps(non_zero_scores))
            return reward_events, grouped_val_score_responses
        except Exception as e:
            error_message = f"Search Summary Relevance get_rewards: {str(e)}"
            tb_str = traceback.format_exception(type(e), e, e.__traceback__)
            bt.logging.error("\n".join(tb_str) + error_message)
            reward_events = []
            for response in responses:
                reward_event = BaseRewardEvent()
                reward_event.reward = 0
                reward_events.append(reward_event)
            return reward_events, {}
