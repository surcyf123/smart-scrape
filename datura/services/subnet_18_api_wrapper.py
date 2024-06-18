import random
import pydantic
import bittensor as bt
from typing import AsyncIterator, Dict, List
from starlette.responses import StreamingResponse


class Subnet18:
    def __init__(self, wallet):
        self.wallet = wallet
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.metagraph = bt.metagraph(netuid=18)
        self.top_miners_to_use = 100
        self.validator_uid = self.metagraph.hotkeys.index(wallet.hotkey.ss58_address)
        self.axon_to_use = self.metagraph.axons[self.validator_uid]

    async def handle_response(self, responses):
        full_response = ""
        try:
            for resp in responses:
                async for chunk in resp:
                    if isinstance(chunk, str):
                        full_response += chunk
                        print(chunk)
                    else:
                        print(f"chunk is not a str: {chunk}")
        except Exception as e:
            print(f"Error processing response for uid {e}")
        return full_response

    async def query_miner(
        self,
        dendrite,
        axon_to_use,
        synapse: bt.StreamingSynapse,
        timeout: int,
        streaming: bool,
    ):
        try:
            responses = dendrite.query(
                axons=[axon_to_use],
                synapse=synapse,
                deserialize=False,
                timeout=timeout,
                streaming=streaming,
            )
            return await self.handle_response(responses)
        except Exception as e:
            bt.logging.error(f"Error occurred during querying miner: {e}")
            return None

    def get_random_miner_uid(self):
        top_miner_uids = self.metagraph.I.argsort(descending=True)[
            : self.top_miners_to_use
        ]
        return random.choice(top_miner_uids)

    def get_miner_uid(self, previous: int = 0):
        top_miner_uids = self.metagraph.I.argsort(descending=True)[
            : self.top_miners_to_use
        ]
        current = previous + 1
        return (top_miner_uids[current], current)

    async def query(
        self,
        messages,
        provider="OpenAI",
        model="gpt-3.5-turbo-16k",
        miner_uid=None,
        temperature=0.2,
    ):
        try:
            bt.logging.info("Calling query of Subnet 18")
            if not miner_uid:
                bt.logging.error("Miner UID is not provided for querying subnet 18")
                return

            synapse = StreamPrompting(
                messages=messages,
                provider=provider,
                model=model,
                uid=miner_uid,
                completion="",
                temperature=temperature,
            )
            return await self.query_miner(
                self.dendrite,
                self.axon_to_use,
                synapse,
                60,  # timeout
                True,  # streaming
            )
        except Exception as e:
            bt.logging.error(f"Error occurred during querying with subnet 18: {e}")
            return None


# Clone of synapse of subnet 18
class StreamPrompting(bt.StreamingSynapse):
    @pydantic.root_validator(pre=True)
    def check_messages_content_type(cls, values):
        messages = values.get("messages", [])
        for i, message in enumerate(messages):
            if "content" in message and not isinstance(message["content"], str):
                raise TypeError(
                    f"Message at index {i} has non-string 'content': {message['content']}"
                )
        return values

    messages: List[Dict[str, str]] = pydantic.Field(
        ...,
        title="Messages",
        description="A list of messages in the StreamPrompting scenario, "
        "each containing a role and content. Immutable.",
        allow_mutation=False,
    )

    required_hash_fields: List[str] = pydantic.Field(
        ["messages"],
        title="Required Hash Fields",
        description="A list of required fields for the hash.",
        allow_mutation=False,
    )

    seed: int = pydantic.Field(
        default="1234",
        title="Seed",
        description="Seed for text generation. This attribute is immutable and cannot be updated.",
    )

    temperature: float = pydantic.Field(
        default=0.0001,
        title="Temperature",
        description="Temperature for text generation. "
        "This attribute is immutable and cannot be updated.",
    )

    max_tokens: int = pydantic.Field(
        default=2048,
        title="Max Tokens",
        description="Max tokens for text generation. "
        "This attribute is immutable and cannot be updated.",
    )

    top_p: float = pydantic.Field(
        default=0.001,
        title="Top_p",
        description="Top_p for text generation. The sampler will pick one of "
        "the top p percent tokens in the logit distirbution. "
        "This attribute is immutable and cannot be updated.",
    )

    top_k: int = pydantic.Field(
        default=1,
        title="Top_k",
        description="Top_k for text generation. Sampler will pick one of  "
        "the k most probablistic tokens in the logit distribtion. "
        "This attribute is immutable and cannot be updated.",
    )

    completion: str = pydantic.Field(
        "",
        title="Completion",
        description="Completion status of the current StreamPrompting object. "
        "This attribute is mutable and can be updated.",
    )

    provider: str = pydantic.Field(
        default="OpenAI",
        title="Provider",
        description="The provider to use when calling for your response. "
        "Options: OpenAI, Anthropic, Gemini",
    )

    model: str = pydantic.Field(
        default="gpt-3.5-turbo",
        title="model",
        description="The model to use when calling provider for your response.",
    )

    uid: int = pydantic.Field(
        default=3,
        title="uid",
        description="The UID to send the streaming synapse to",
    )

    timeout: int = pydantic.Field(
        default=60,
        title="timeout",
        description="The timeout for the dendrite of the streaming synapse",
    )

    streaming: bool = pydantic.Field(
        default=True,
        title="streaming",
        description="whether to stream the output",
    )

    async def process_streaming_response(
        self, response: StreamingResponse
    ) -> AsyncIterator[str]:
        if self.completion is None:
            self.completion = ""
        async for chunk in response.content.iter_any():
            tokens = chunk.decode("utf-8")
            for token in tokens:
                if token:
                    self.completion += token
            yield tokens

    def deserialize(self) -> str:
        return self.completion

    def extract_response_json(self, response: StreamingResponse) -> dict:
        headers = {
            k.decode("utf-8"): v.decode("utf-8")
            for k, v in response.__dict__["_raw_headers"]
        }

        def extract_info(prefix: str) -> dict[str, str]:
            return {
                key.split("_")[-1]: value
                for key, value in headers.items()
                if key.startswith(prefix)
            }

        return {
            "name": headers.get("name", ""),
            "timeout": float(headers.get("timeout", 0)),
            "total_size": int(headers.get("total_size", 0)),
            "header_size": int(headers.get("header_size", 0)),
            "dendrite": extract_info("bt_header_dendrite"),
            "axon": extract_info("bt_header_axon"),
            "messages": self.messages,
            "completion": self.completion,
        }
