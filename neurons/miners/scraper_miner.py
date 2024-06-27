import traceback
import bittensor as bt
from starlette.types import Send
from datura.dataset.tool_return import response_order_from_str
from datura.protocol import (
    ScraperStreamingSynapse,
)
from datura.tools.tool_manager import ToolManager
from datura.dataset.date_filters import (
    DateFilter,
    DateFilterType,
    get_specified_date_filter,
)
from datetime import datetime
import pytz


class ScraperMiner:
    def __init__(self, miner: any):
        self.miner = miner

    async def smart_scraper(self, synapse: ScraperStreamingSynapse, send: Send):
        try:
            model = synapse.model
            prompt = synapse.messages
            seed = synapse.seed
            tools = synapse.tools
            is_intro_text = synapse.is_intro_text
            response_order = response_order_from_str(synapse.response_order)

            bt.logging.trace(synapse)

            bt.logging.info(
                "================================== Prompt ==================================="
            )
            bt.logging.info(prompt)
            bt.logging.info(
                "================================== Prompt ===================================="
            )

            date_filter = get_specified_date_filter(DateFilterType.PAST_2_WEEKS)

            if synapse.start_date and synapse.end_date and synapse.date_filter_type:
                start_date = datetime.strptime(
                    synapse.start_date, "%Y-%m-%dT%H:%M:%SZ"
                ).replace(tzinfo=pytz.utc)

                end_date = datetime.strptime(
                    synapse.end_date, "%Y-%m-%dT%H:%M:%SZ"
                ).replace(tzinfo=pytz.utc)

                date_filter = DateFilter(
                    start_date=start_date,
                    end_date=end_date,
                    date_filter_type=DateFilterType(synapse.date_filter_type),
                )

            tool_manager = ToolManager(
                prompt=prompt,
                manual_tool_names=tools,
                send=send,
                is_intro_text=is_intro_text,
                miner=self.miner,
                language=synapse.language,
                region=synapse.region,
                date_filter=date_filter,
                google_date_filter=synapse.google_date_filter,
                response_order=response_order,
            )

            await tool_manager.run()

            bt.logging.info("End of Streaming")

        except Exception as e:
            bt.logging.error(f"error in twitter scraper {e}\n{traceback.format_exc()}")
