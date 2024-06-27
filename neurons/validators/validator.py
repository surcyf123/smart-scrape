import torch
import wandb
import asyncio
import concurrent
import traceback
import random
import copy
import bittensor as bt
import time
import sys
from datura.protocol import IsAlive
from neurons.validators.scraper_validator import ScraperValidator
from config import add_args, check_config, config
from weights import init_wandb, set_weights, get_weights
from traceback import print_exception
from base_validator import AbstractNeuron
from datura import QUERY_MINERS
from datura.misc import ttl_get_block
from datura.utils import resync_metagraph, save_logs_in_chunks


class Neuron(AbstractNeuron):
    @classmethod
    def check_config(cls, config: "bt.Config"):
        check_config(cls, config)

    @classmethod
    def add_args(cls, parser):
        add_args(cls, parser)

    @classmethod
    def config(cls):
        return config(cls)

    subtensor: "bt.subtensor"
    wallet: "bt.wallet"
    metagraph: "bt.metagraph"
    dendrite: "bt.dendrite"

    scraper_validator: "ScraperValidator"
    moving_average_scores: torch.Tensor = None
    uid: int = None
    shutdown_event: asyncio.Event()

    @property
    def block(self):
        return ttl_get_block(self)

    def __init__(self):
        self.config = Neuron.config()
        self.check_config(self.config)
        bt.logging(config=self.config, logging_dir=self.config.neuron.full_path)
        print(self.config)
        bt.logging.info("neuron.__init__()")

        self.initialize_components()

        init_wandb(self)

        self.scraper_validator = ScraperValidator(
            neuron=self,
            wallet=self.wallet
        )
        bt.logging.info("initialized_validators")

        # Init the event loop.
        self.loop = asyncio.get_event_loop()
        self.step = 0
        self.check_registered()

        # Init Weights.
        bt.logging.debug("loading", "moving_averaged_scores")
        self.moving_averaged_scores = torch.zeros((self.metagraph.n)).to(
            self.config.neuron.device
        )
        bt.logging.debug(str(self.moving_averaged_scores))
        self.available_uids = []
        self.thread_executor = concurrent.futures.ThreadPoolExecutor(
            thread_name_prefix="asyncio"
        )
        # Init sync with the network. Updates the metagraph.
        self.sync()

    async def run_sync_in_async(self, fn):
        return await self.loop.run_in_executor(self.thread_executor, fn)

    def initialize_components(self):
        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.info(
            f"Running validator for subnet: {self.config.netuid} on network: {self.config.subtensor.chain_endpoint}"
        )
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(
                f"Your validator: {self.wallet} is not registered to chain connection: {self.subtensor}. Run btcli register --netuid 18 and try again."
            )
            exit()

    async def update_available_uids_periodically(self):
        while True:
            start_time = time.time()
            try:
                # if self.should_sync_metagraph():
                #     resync_metagraph(self)

                # unless it's used elsewhere.
                self.available_uids = await self.get_available_uids_is_alive()
                bt.logging.info(
                    f"Number of available UIDs for periodic update: Amount: {len(self.available_uids)}, UIDs: {self.available_uids}"
                )
            except Exception as e:
                bt.logging.error(
                    f"update_available_uids_periodically Failed to update available UIDs: {e}"
                )
                # Consider whether to continue or break the loop upon certain errors.

            end_time = time.time()
            execution_time = end_time - start_time
            bt.logging.info(
                f"update_available_uids_periodically Execution time for getting available UIDs amount is: {execution_time} seconds"
            )

            await asyncio.sleep(self.config.neuron.update_available_uids_interval)

    async def check_uid(self, axon, uid):
        """Asynchronously check if a UID is available."""
        try:
            response = await self.dendrite(
                axon, IsAlive(), deserialize=False, timeout=15
            )
            if response.is_success:
                bt.logging.trace(f"UID {uid} is active")
                return axon  # Return the axon info instead of the UID
            else:
                raise Exception(f"Is not active")
        except Exception as e:
            bt.logging.trace(f"Checking UID {uid}: {e}\n{traceback.format_exc()}")
            raise e

    async def get_available_uids_is_alive(self):
        """Get a dictionary of available UIDs and their axons asynchronously."""
        tasks = {
            uid.item(): self.check_uid(
                self.metagraph.axons[uid.item()],
                uid.item(),
            )
            for uid in self.metagraph.uids
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        # Filter out the exceptions and keep the successful results
        available_uids = [
            uid
            for uid, result in zip(tasks.keys(), results)
            if not isinstance(result, Exception)
        ]

        return available_uids

    async def get_uids(
        self,
        strategy=QUERY_MINERS.RANDOM,
        is_only_allowed_miner=False,
        specified_uids=None,
    ):
        if len(self.available_uids) == 0:
            bt.logging.info("No available UIDs, attempting to refresh list.")
            return self.available_uids

        # Filter uid_list based on specified_uids and only_allowed_miners
        uid_list = [
            uid
            for uid in self.available_uids
            if (not specified_uids or uid in specified_uids)
            and (
                not is_only_allowed_miner
                or self.metagraph.axons[uid].coldkey
                in self.config.neuron.only_allowed_miners
            )
        ]

        if strategy == QUERY_MINERS.RANDOM:
            uids = (
                torch.tensor([random.choice(uid_list)])
                if uid_list
                else torch.tensor([])
            )
        elif strategy == QUERY_MINERS.ALL:
            uids = torch.tensor(uid_list) if uid_list else torch.tensor([])
        bt.logging.info(f"Run uids ---------- Amount: {len(uids)} | {uids}")
        # uid_list = list(available_uids.keys())
        return uids.to(self.config.neuron.device)

    async def update_scores(
        self,
        wandb_data,
        prompt,
        responses,
        uids,
        rewards,
        all_rewards,
        all_original_rewards,
        val_score_responses_list,
        neuron,
    ):
        try:
            if self.config.wandb_on:
                wandb.log(wandb_data)

            asyncio.create_task(
                save_logs_in_chunks(
                    self,
                    prompt=prompt,
                    responses=responses,
                    uids=uids,
                    rewards=rewards,
                    summary_rewards=all_rewards[0],
                    twitter_rewards=all_rewards[1],
                    search_rewards=all_rewards[2],
                    original_summary_rewards=all_original_rewards[0],
                    original_twitter_rewards=all_original_rewards[1],
                    original_search_rewards=all_original_rewards[2],
                    tweet_scores=val_score_responses_list[1],
                    search_scores=val_score_responses_list[2],
                    weights=get_weights(self),
                    neuron=neuron,
                    netuid=self.config.netuid,
                )
            )
        except Exception as e:
            bt.logging.error(f"Error in update_scores: {e}")
            raise e

    def update_moving_averaged_scores(self, uids, rewards):
        try:
            # Ensure uids is a tensor
            if not isinstance(uids, torch.Tensor):
                uids = torch.tensor(
                    uids, dtype=torch.long, device=self.config.neuron.device
                )

            # Ensure rewards is also a tensor and on the correct device
            if not isinstance(rewards, torch.Tensor):
                rewards = torch.tensor(rewards, device=self.config.neuron.device)

            scattered_rewards = self.moving_averaged_scores.scatter(
                0, uids, rewards
            ).to(self.config.neuron.device)
            average_reward = torch.mean(scattered_rewards)
            bt.logging.info(
                f"Scattered reward: {average_reward:.6f}"
            )  # Rounds to 6 decimal places for logging

            alpha = self.config.neuron.moving_average_alpha
            self.moving_averaged_scores = alpha * scattered_rewards + (
                1 - alpha
            ) * self.moving_averaged_scores.to(self.config.neuron.device)
            bt.logging.info(
                f"Moving averaged scores: {torch.mean(self.moving_averaged_scores):.6f}"
            )  # Rounds to 6 decimal places for logging
            return scattered_rewards
        except Exception as e:
            bt.logging.error(f"Error in update_moving_averaged_scores: {e}")
            raise e

    async def query_synapse(self, strategy=QUERY_MINERS.RANDOM):
        try:
            # self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)
            await self.scraper_validator.query_and_score(strategy)
        except Exception as e:
            bt.logging.error(f"General exception: {e}\n{traceback.format_exc()}")
            await asyncio.sleep(100)

    async def run_synthetic_queries(self, strategy=QUERY_MINERS.RANDOM):
        bt.logging.info(f"Starting run_synthetic_queries with strategy={strategy}")
        total_start_time = time.time()
        try:

            async def run_forward():
                start_time = time.time()
                bt.logging.info(
                    f"Running step forward for query_synapse, Step: {self.step}"
                )
                coroutines = [self.query_synapse(strategy) for _ in range(1)]
                await asyncio.gather(*coroutines)
                end_time = time.time()
                bt.logging.info(
                    f"Completed gathering coroutines for query_synapse in {end_time - start_time:.2f} seconds"
                )

            bt.logging.info("Running coroutines with run_until_complete")
            self.loop.run_until_complete(run_forward())
            bt.logging.info("Completed running coroutines with run_until_complete")

            sync_start_time = time.time()
            bt.logging.info("Calling sync method")
            self.sync()
            bt.logging.info("Completed calling sync method")

            sync_end_time = time.time()
            bt.logging.info(
                f"Sync method execution time: {sync_end_time - sync_start_time:.2f} seconds"
            )

            self.step += 1
            bt.logging.info(f"Incremented step to {self.step}")
        except Exception as err:
            bt.logging.error("Error in run_synthetic_queries", str(err))
            bt.logging.debug(print_exception(type(err), err, err.__traceback__))
        finally:
            total_end_time = time.time()
            total_execution_time = (total_end_time - total_start_time) / 60
            bt.logging.info(
                f"Total execution time for run_synthetic_queries: {total_execution_time:.2f} minutes"
            )

    def sync(self):
        """
        Wrapper for synchronizing the state of the network for the given miner or validator.
        """
        # Ensure miner or validator hotkey is still registered on the network.
        self.check_registered()

        if self.should_sync_metagraph():
            bt.logging.info("Syncing metagraph as per condition.")
            resync_metagraph(self)
        else:
            bt.logging.info("No need to sync metagraph at this moment.")

        if self.should_set_weights():
            weight_set_start_time = time.time()
            bt.logging.info("Setting weights as per condition.")
            set_weights(self)
            weight_set_end_time = time.time()
            bt.logging.info(
                f"Weight setting execution time: {weight_set_end_time - weight_set_start_time:.2f} seconds"
            )
        else:
            bt.logging.info("No need to set weights at this moment.")

    def check_registered(self):
        # --- Check for registration.
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            bt.logging.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again"
            )
            sys.exit()

    def should_sync_metagraph(self):
        """
        Check if enough epoch blocks have elapsed since the last checkpoint to sync.
        """
        difference = self.block - self.metagraph.last_update[self.uid]
        print(
            f"Current block: {self.block}, Last update for UID {self.uid}: {self.metagraph.last_update[self.uid]}, Difference: {difference}"
        )
        should_set = difference > self.config.neuron.checkpoint_block_length
        bt.logging.info(f"Should set weights: {should_set}")
        # return should_set
        return True  # Update right not based on interval of synthetic data

    def should_set_weights(self) -> bool:
        # Don't set weights on initialization.
        # if self.step == 0:
        #     bt.logging.info("Skipping weight setting on initialization.")
        #     return False

        # Check if enough epoch blocks have elapsed since the last epoch.
        if self.config.neuron.disable_set_weights:
            bt.logging.info("Weight setting is disabled by configuration.")
            return False

        # Define appropriate logic for when set weights.
        difference = self.block - self.metagraph.last_update[self.uid]
        print(
            f"Current block: {self.block}, Last update for UID {self.uid}: {self.metagraph.last_update[self.uid]}, Difference: {difference}"
        )
        should_set = difference > self.config.neuron.checkpoint_block_length
        bt.logging.info(f"Should set weights: {should_set}")
        # return should_set
        return True  # Update right not based on interval of synthetic data

    async def run(self):
        await asyncio.sleep(10)
        self.sync()
        self.loop.create_task(self.update_available_uids_periodically())
        bt.logging.info(f"Validator starting at block: {self.block}")

        try:

            async def run_with_interval(interval, strategy):
                query_count = 0  # Initialize query count
                while True:
                    try:
                        if not self.available_uids:
                            bt.logging.info(
                                "No available UIDs, sleeping for 10 seconds."
                            )
                            await asyncio.sleep(10)
                            continue
                        self.loop.create_task(self.run_synthetic_queries(strategy))

                        await asyncio.sleep(
                            interval
                        )  # Wait for 1800 seconds (30 minutes)
                    except Exception as e:
                        bt.logging.error(f"Error during task execution: {e}")
                        await asyncio.sleep(interval)  # Wait before retrying

            if self.config.neuron.run_random_miner_syn_qs_interval > 0:
                self.loop.create_task(
                    run_with_interval(
                        self.config.neuron.run_all_miner_syn_qs_interval,
                        QUERY_MINERS.RANDOM,
                    )
                )

            if self.config.neuron.run_all_miner_syn_qs_interval > 0:
                self.loop.create_task(
                    run_with_interval(
                        self.config.neuron.run_all_miner_syn_qs_interval,
                        QUERY_MINERS.ALL,
                    )
                )
                # If someone intentionally stops the validator, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Validator killed by keyboard interrupt.")
            sys.exit()

        # In case of unforeseen errors, the validator will log the error and quit
        except Exception as err:
            bt.logging.error("Error during validation", str(err))
            bt.logging.debug(print_exception(type(err), err, err.__traceback__))
            self.should_exit = True


def main():
    asyncio.run(Neuron().run())


if __name__ == "__main__":
    main()
