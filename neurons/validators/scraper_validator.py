import math
import torch
import wandb
import random
import json
import bittensor as bt
from base_validator import AbstractNeuron
from datura.protocol import (
    ScraperStreamingSynapse,
    TwitterPromptAnalysisResult,
    SearchSynapse,
)
from datura.stream import process_async_responses, process_single_response
from reward import RewardModelType, RewardScoringType
from typing import List
from utils.mock import MockRewardModel
import time
from neurons.validators.penalty import (
    TaskValidationPenaltyModel,
    AccuracyPenaltyModel,
    LinkValidationPenaltyModel,
)
from neurons.validators.reward.summary_relevance import SummaryRelevanceRewardModel
from neurons.validators.reward.twitter_content_relevance import (
    TwitterContentRelevanceModel,
)
from neurons.validators.reward.search_content_relevance import (
    WebSearchContentRelevanceModel,
)
from neurons.validators.reward.performance_reward import (
    PerformanceRewardModel,
)
from neurons.validators.reward.reward_llm import RewardLLM
from neurons.validators.utils.tasks import TwitterTask, SearchTask

from datura.dataset import MockTwitterQuestionsDataset, QuestionsDataset
from datura.services.twitter_api_wrapper import TwitterAPIClient
from datura import QUERY_MINERS
import asyncio
from aiostream import stream
from datura.dataset.date_filters import (
    get_random_date_filter,
    get_specified_date_filter,
    DateFilterType,
)


class ScraperValidator:
    def __init__(self, neuron: AbstractNeuron, wallet: bt.wallet):
        self.streaming = True
        self.query_type = "text"
        self.model = "gpt-3.5-turbo-0125"
        self.weight = 1
        self.seed = 1234
        self.neuron = neuron
        self.timeout = 180
        self.tools = [
            ["Twitter Search", "Reddit Search"],
            ["Twitter Search", "Reddit Search", "Hacker News Search"],
            ["Twitter Search", "Google Search"],
            ["Twitter Search", "Google Search"],
            ["Twitter Search", "Hacker News Search"],
            ["Twitter Search", "Wikipedia Search", "ArXiv Search"],
            ["Twitter Search", "Youtube Search"],
            ["Twitter Search", "Youtube Search"],
            ["Twitter Search", "Google News Search"],
            ["Twitter Search", "Reddit Search"],
            ["Twitter Search", "Reddit Search"],
            ["Twitter Search", "Hacker News Search"],
            ["Twitter Search", "ArXiv Search"],
            ["Twitter Search", "Wikipedia Search"],
            ["Twitter Search", "Google Search"],
            ["Twitter Search", "Google News Search"],
        ]
        self.language = "en"
        self.region = "us"
        self.date_filter = "qdr:w"  # Past week
        self.max_tools_result_amount = 10
        self.wallet = wallet

        # Init device.
        bt.logging.debug("loading", "device")
        bt.logging.debug(
            "self.neuron.config.neuron.device = ", str(self.neuron.config.neuron.device)
        )

        self.reward_weights = torch.tensor(
            [
                self.neuron.config.reward.summary_relevance_weight,
                self.neuron.config.reward.twitter_content_weight,
                self.neuron.config.reward.web_search_relavance_weight,
                self.neuron.config.reward.performance_weight,
            ],
            dtype=torch.float32,
        ).to(self.neuron.config.neuron.device)

        if self.reward_weights.sum() != 1:
            message = (
                f"Reward function weights do not sum to 1 (Current sum: {self.reward_weights.sum()}.)"
                f"Check your reward config file at `reward/config.py` or ensure that all your cli reward flags sum to 1."
            )
            bt.logging.error(message)
            raise Exception(message)

        self.reward_llm = RewardLLM(wallet=self.wallet)
        # if (
        #     self.neuron.config.reward.twitter_content_weight > 0
        #     or self.neuron.config.reward.summary_relevance_weight > 0
        # ) and not self.neuron.config.neuron.is_disable_tokenizer_reward:
        #     self.reward_llm.init_pipe_zephyr()

        self.reward_functions = [
            (
                SummaryRelevanceRewardModel(
                    device=self.neuron.config.neuron.device,
                    scoring_type=RewardScoringType.summary_relevance_score_template,
                    llm_reward=self.reward_llm,
                )
                if self.neuron.config.reward.summary_relevance_weight > 0
                else MockRewardModel(RewardModelType.summary_relavance_match.value)
            ),
            (
                TwitterContentRelevanceModel(
                    device=self.neuron.config.neuron.device,
                    scoring_type=RewardScoringType.summary_relevance_score_template,
                    llm_reward=self.reward_llm,
                )
                if self.neuron.config.reward.twitter_content_weight > 0
                else MockRewardModel(RewardModelType.twitter_content_relevance.value)
            ),
            (
                WebSearchContentRelevanceModel(
                    device=self.neuron.config.neuron.device,
                    scoring_type=RewardScoringType.search_relevance_score_template,
                    llm_reward=self.reward_llm,
                )
                if self.neuron.config.reward.web_search_relavance_weight > 0
                else MockRewardModel(RewardModelType.search_content_relevance.value)
            ),
            (
                PerformanceRewardModel(
                    device=self.neuron.config.neuron.device,
                )
                if self.neuron.config.reward.performance_weight > 0
                else MockRewardModel(RewardModelType.performance_score.value)
            ),
        ]

        self.penalty_functions = [
            # LinkValidationPenaltyModel(max_penalty=0.7),
            # AccuracyPenaltyModel(max_penalty=1),
        ]
        self.twitter_api = TwitterAPIClient()

    async def run_task_and_score(
        self,
        task: TwitterTask,
        strategy=QUERY_MINERS.RANDOM,
        is_only_allowed_miner=True,
        is_intro_text=False,
        specified_uids=None,
        date_filter=None,
        tools=[],
        language="en",
        region="us",
        google_date_filter="qdr:w",
    ):
        task_name = task.task_name
        prompt = task.compose_prompt()

        bt.logging.debug("run_task", task_name)

        # Record event start time.
        event = {"name": task_name, "task_type": task.task_type}
        start_time = time.time()

        # Get random id on that step
        uids = await self.neuron.get_uids(
            strategy=strategy,
            is_only_allowed_miner=is_only_allowed_miner,
            specified_uids=specified_uids,
        )

        start_date = date_filter.start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_date = date_filter.end_date.strftime("%Y-%m-%dT%H:%M:%SZ")

        axons = [self.neuron.metagraph.axons[uid] for uid in uids]
        synapse = ScraperStreamingSynapse(
            messages=prompt,
            model=self.model,
            seed=self.seed,
            is_intro_text=is_intro_text,
            start_date=start_date,
            end_date=end_date,
            date_filter_type=date_filter.date_filter_type.value,
            tools=tools,
            language=language,
            region=region,
            google_date_filter=google_date_filter,
        )

        # Make calls to the network with the prompt.
        async_responses = await self.neuron.dendrite.forward(
            axons=axons,
            synapse=synapse,
            timeout=self.timeout,
            streaming=self.streaming,
            deserialize=False,
        )

        return async_responses, uids, event, start_time

    async def compute_rewards_and_penalties(
        self, event, prompt, task, responses, uids, start_time
    ):
        try:
            if not len(uids):
                bt.logging.warning("No UIDs provided for logging event.")
                return

            bt.logging.info("Computing rewards and penalties")

            rewards = torch.zeros(len(responses), dtype=torch.float32).to(
                self.neuron.config.neuron.device
            )

            all_rewards = []
            all_original_rewards = []
            val_score_responses_list = []

            for weight_i, reward_fn_i in zip(
                self.reward_weights, self.reward_functions
            ):
                start_time = time.time()
                (
                    reward_i_normalized,
                    reward_event,
                    val_score_responses,
                    original_rewards,
                ) = reward_fn_i.apply(task.base_text, responses, task.task_name, uids)

                all_rewards.append(reward_i_normalized)
                all_original_rewards.append(original_rewards)
                val_score_responses_list.append(val_score_responses)

                rewards += weight_i * reward_i_normalized.to(
                    self.neuron.config.neuron.device
                )
                if not self.neuron.config.neuron.disable_log_rewards:
                    event = {**event, **reward_event}
                execution_time = time.time() - start_time
                bt.logging.trace(str(reward_fn_i.name), reward_i_normalized.tolist())
                bt.logging.info(
                    f"Applied reward function: {reward_fn_i.name} in {execution_time / 60:.2f} minutes"
                )

            for penalty_fn_i in self.penalty_functions:
                raw_penalty_i, adjusted_penalty_i, applied_penalty_i = (
                    penalty_fn_i.apply_penalties(responses, task)
                )
                penalty_start_time = time.time()
                rewards *= applied_penalty_i.to(self.neuron.config.neuron.device)
                penalty_execution_time = time.time() - penalty_start_time
                if not self.neuron.config.neuron.disable_log_rewards:
                    event[penalty_fn_i.name + "_raw"] = raw_penalty_i.tolist()
                    event[penalty_fn_i.name + "_adjusted"] = adjusted_penalty_i.tolist()
                    event[penalty_fn_i.name + "_applied"] = applied_penalty_i.tolist()
                bt.logging.trace(str(penalty_fn_i.name), applied_penalty_i.tolist())
                bt.logging.info(
                    f"Applied penalty function: {penalty_fn_i.name} in {penalty_execution_time:.2f} seconds"
                )

            scattered_rewards = self.neuron.update_moving_averaged_scores(uids, rewards)
            self.log_event(
                task, event, start_time, uids, rewards, prompt=task.compose_prompt()
            )

            scores = torch.zeros(len(self.neuron.metagraph.hotkeys))
            uid_scores_dict = {}
            wandb_data = {
                "modality": "twitter_scrapper",
                "prompts": {},
                "responses": {},
                "scores": {},
                "timestamps": {},
                "summary_reward": {},
                "twitter_reward": {},
                "search_reward": {},
                "latency_reward": {},
            }
            bt.logging.info(
                f"======================== Reward ==========================="
            )
            # Initialize an empty list to accumulate log messages
            log_messages = []
            for uid_tensor, reward, response in zip(uids, rewards.tolist(), responses):
                uid = uid_tensor.item()
                completion_length = (
                    len(response.completion) if response.completion is not None else 0
                )
                completion_links_length = (
                    len(response.completion_links)
                    if response.completion_links is not None
                    else 0
                )
                # Accumulate log messages instead of logging them immediately
                log_messages.append(
                    f"UID: {uid}, R: {round(reward, 3)}, C: {completion_length}, L: {completion_links_length}"
                )
                bt.logging.trace(f"{response.completion}")

            # Log the accumulated messages in groups of three
            for i in range(0, len(log_messages), 3):
                bt.logging.info(" | ".join(log_messages[i : i + 3]))

            bt.logging.info(
                f"======================== Reward ==========================="
            )

            summary_rewards = all_rewards[0]
            twitter_rewards = all_rewards[1]
            search_rewards = all_rewards[2]
            latency_rewards = all_rewards[3]
            zipped_rewards = zip(
                uids,
                rewards.tolist(),
                responses,
                summary_rewards,
                twitter_rewards,
                search_rewards,
                latency_rewards,
            )

            for (
                uid_tensor,
                reward,
                response,
                summary_reward,
                twitter_reward,
                search_reward,
                latency_reward,
            ) in zipped_rewards:
                uid = uid_tensor.item()  # Convert tensor to int
                uid_scores_dict[uid] = reward
                scores[uid] = reward  # Now 'uid' is an int, which is a valid key type
                wandb_data["scores"][uid] = reward
                wandb_data["responses"][uid] = response.completion
                wandb_data["prompts"][uid] = prompt
                wandb_data["summary_reward"][uid] = summary_reward
                wandb_data["twitter_reward"][uid] = twitter_reward
                wandb_data["search_reward"][uid] = search_reward
                wandb_data["latency_reward"][uid] = latency_reward

            await self.neuron.update_scores(
                wandb_data=wandb_data,
                prompt=prompt,
                responses=responses,
                uids=uids,
                rewards=rewards,
                all_rewards=all_rewards,
                all_original_rewards=all_original_rewards,
                val_score_responses_list=val_score_responses_list,
                neuron=self.neuron,
            )

            return rewards, uids, val_score_responses_list, event
        except Exception as e:
            bt.logging.error(f"Error in compute_rewards_and_penalties: {e}")
            raise e

    def log_event(self, task, event, start_time, uids, rewards, prompt):
        event.update(
            {
                "step_length": time.time() - start_time,
                "prompt": prompt,
                "uids": uids.tolist(),
                "rewards": rewards.tolist(),
                "propmt": task.base_text,
            }
        )
        bt.logging.debug("Run Task event:", str(event))

    async def process_async_responses(async_responses):
        tasks = [resp for resp in async_responses]
        responses = await asyncio.gather(*tasks)
        for response in responses:
            stream_text = "".join([chunk[1] for chunk in response if not chunk[0]])
            if stream_text:
                yield stream_text  # Yield stream text as soon as it's available
            # Instead of returning, yield final synapse objects with a distinct flag
            final_synapse = next((chunk[1] for chunk in response if chunk[0]), None)
            if final_synapse:
                yield (True, final_synapse)  # Yield final synapse with a flag

    async def query_and_score(self, strategy=QUERY_MINERS.RANDOM):
        try:
            dataset = QuestionsDataset()
            tools = random.choice(self.tools)
            prompt = await dataset.generate_new_question_with_openai(tools)

            task_name = "augment"
            task = TwitterTask(
                base_text=prompt,
                task_name=task_name,
                task_type="twitter_scraper",
                criteria=[],
            )

            if not len(self.neuron.available_uids):
                bt.logging.info("No available UIDs, skipping task execution.")
                return

            async_responses, uids, event, start_time = await self.run_task_and_score(
                task=task,
                strategy=strategy,
                is_only_allowed_miner=False,
                date_filter=get_random_date_filter(),
                tools=tools,
                language=self.language,
                region=self.region,
                google_date_filter=self.date_filter,
            )

            final_synapses = []
            async for value in process_async_responses(async_responses):
                if isinstance(value, bt.Synapse):
                    final_synapses.append(value)
                else:
                    pass

            await self.compute_rewards_and_penalties(
                event=event,
                prompt=prompt,
                task=task,
                responses=final_synapses,
                uids=uids,
                start_time=start_time,
            )
        except Exception as e:
            bt.logging.error(f"Error in query_and_score: {e}")
            raise e

    async def organic(self, query):
        try:
            prompt = query["content"]
            tools = query.get("tools", [])
            date_filter_type = query.get("date_filter", DateFilterType.PAST_WEEK.value)
            date_filter_type = DateFilterType(date_filter_type)

            task_name = "augment"
            task = TwitterTask(
                base_text=prompt,
                task_name=task_name,
                task_type="twitter_scraper",
                criteria=[],
            )

            if not len(self.neuron.available_uids):
                bt.logging.info("Not available uids")
                raise StopAsyncIteration("Not available uids")

            date_filter = get_specified_date_filter(date_filter_type)

            async_responses, uids, event, start_time = await self.run_task_and_score(
                task=task,
                strategy=QUERY_MINERS.RANDOM,
                is_only_allowed_miner=True,
                is_intro_text=True,
                tools=tools,
                language=self.language,
                region=self.region,
                date_filter=date_filter,
                google_date_filter=self.date_filter,
            )
            final_synapses = []
            for response in async_responses:
                async for value in response:
                    if isinstance(value, bt.Synapse):
                        final_synapses.append(value)
                    else:
                        yield value

            async def process_and_score_responses():
                await self.compute_rewards_and_penalties(
                    event=event,
                    prompt=prompt,
                    task=task,
                    responses=final_synapses,
                    uids=uids,
                    start_time=start_time,
                )

            asyncio.create_task(process_and_score_responses())
        except Exception as e:
            bt.logging.error(f"Error in organic: {e}")
            raise e

    def format_val_score_responses(self, val_score_responses_list):
        formatted_scores = []
        for response_dict in val_score_responses_list:
            if response_dict:  # Check if the dictionary is not empty
                formatted_scores.append(json.dumps(response_dict, indent=4))
            else:
                formatted_scores.append("{}")  # Empty dictionary
        return "\n".join(formatted_scores)

    async def organic_specified(self, query, specified_uids=None):
        def format_response(uid, text):
            return json.dumps(
                {"uid": uid, "type": "text", "content": text, "role": text}
            )

        try:
            prompt = query["content"]
            tools = query.get("tools", [])
            # tools = ["Google Search", "Youtube Search"]
            # tools = ["Twitter Search"]
            date_filter_type = query.get(
                "date_filter", DateFilterType.PAST_2_DAYS.value
            )
            date_filter_type = DateFilterType(date_filter_type)

            task_name = "augment"
            task = TwitterTask(
                base_text=prompt,
                task_name=task_name,
                task_type="twitter_scraper",
                criteria=[],
            )

            if not len(self.neuron.available_uids):
                bt.logging.info("Not available uids")
                raise StopAsyncIteration("Not available uids")

            date_filter = get_specified_date_filter(date_filter_type)

            async_responses, uids, event, start_time = await self.run_task_and_score(
                task=task,
                strategy=QUERY_MINERS.ALL,
                is_only_allowed_miner=False,
                specified_uids=specified_uids,
                tools=tools,
                language=self.language,
                region=self.region,
                date_filter=date_filter,
                google_date_filter=self.date_filter,
            )

            async def stream_response(uid, async_response):
                yield format_response(uid, f"\n\nMiner UID {uid}\n")
                yield format_response(
                    uid, "----------------------------------------\n\n"
                )

                async for value in async_response:
                    if isinstance(value, bt.Synapse):
                        yield value
                    else:
                        yield json.dumps({"uid": uid, **json.loads(value)})

            async_responses_with_uid = [
                stream_response(uid.item(), response)
                for uid, response in zip(uids, async_responses)
            ]

            if len(async_responses_with_uid) > 0:
                merged_stream_with_uid = stream.merge(*async_responses_with_uid)

                final_synapses = []

                async with merged_stream_with_uid.stream() as streamer:
                    async for value in streamer:
                        if isinstance(value, bt.Synapse):
                            final_synapses.append(value)
                        else:
                            yield value

                for uid_tensor, response in zip(uids, final_synapses):
                    uid = uid_tensor.item()
                    yield format_response(
                        uid, "\n\n----------------------------------------\n"
                    )
                    yield format_response(
                        uid, f"Scoring Miner UID {uid}. Please wait for the score...\n"
                    )

                start_compute_time = time.time()

                rewards_task = asyncio.create_task(
                    self.compute_rewards_and_penalties(
                        event=event,
                        prompt=prompt,
                        task=task,
                        responses=final_synapses,
                        uids=uids,
                        start_time=start_time,
                    )
                )

                while not rewards_task.done():
                    await asyncio.sleep(
                        30
                    )  # Check every 30 seconds if the task is done
                    elapsed_time = time.time() - start_compute_time
                    if elapsed_time > 60:  # If more than one minute has passed
                        yield f"Waiting for reward scoring... {elapsed_time // 60} minutes elapsed.\n\n"
                        start_compute_time = time.time()  # Reset the timer

                rewards, uids, val_score_responses_list, event = await rewards_task
                for i, uid_tensor in enumerate(uids):
                    uid = uid_tensor.item()
                    reward = rewards[i].item()
                    response = final_synapses[i]

                    # val_score_response = self.format_val_score_responses([val_score_responses_list[i]])

                    tweet_details = val_score_responses_list[1][i]
                    web_details = val_score_responses_list[2][i]

                    search_content_relevance = event.get(
                        "search_content_relevance", [None]
                    )[i]
                    twitter_content_relevance = event.get(
                        "twitter_content_relevance", [None]
                    )[i]
                    summary_relevance = event.get("summary_relavance_match", [None])[i]

                    yield format_response(
                        uid, "----------------------------------------\n\n\n"
                    )
                    yield format_response(
                        uid, f"Miner UID {uid} Reward: {reward:.2f}\n\n\n"
                    )
                    yield format_response(
                        uid, f"Summary score: {summary_relevance:.4f}\n\n\n"
                    )
                    yield format_response(
                        uid, f"Twitter Score: {twitter_content_relevance:.4f}\n\n\n"
                    )
                    yield format_response(
                        uid, f"Web Score: {search_content_relevance}\n\n\n"
                    )
                    yield format_response(uid, f"Tweet details: {tweet_details}\n\n\n")
                    yield format_response(uid, f"Web details: {web_details}\n\n\n")

                missing_uids = set(specified_uids) - set(uid.item() for uid in uids)
                for missing_uid in missing_uids:
                    yield format_response(
                        missing_uid, f"No response from Miner ID: {missing_uid}\n"
                    )
        except Exception as e:
            bt.logging.error(f"Error in query_and_score: {e}")
            raise e

    async def search(self, query: str, tools: List[str], uid: int = None):
        try:
            task_name = "search"

            task = SearchTask(
                base_text=query,
                task_name=task_name,
                task_type="search",
                criteria=[],
            )

            if not len(self.neuron.available_uids):
                bt.logging.info("Not available uids")
                raise StopAsyncIteration("Not available uids")

            prompt = task.compose_prompt()

            bt.logging.debug("run_task", task_name)

            # If uid is not provided, get random uids
            if uid is None:
                uids = await self.neuron.get_uids(
                    strategy=QUERY_MINERS.RANDOM,
                    is_only_allowed_miner=False,
                    specified_uids=None,
                )

                if uids:
                    uid = uids[0]
                else:
                    raise StopAsyncIteration("No available uids")

            axon = self.neuron.metagraph.axons[uid]

            synapse = SearchSynapse(
                query=prompt,
                tools=tools,
                results={},
            )

            synapse: SearchSynapse = await self.neuron.dendrite.call(
                target_axon=axon,
                synapse=synapse,
                timeout=self.timeout,
                deserialize=False,
            )

            return synapse.results
        except Exception as e:
            bt.logging.error(f"Error in search: {e}")
            raise e
