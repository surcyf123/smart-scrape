from enum import Enum


class ResponseOrder(Enum):
    LINKS_FIRST = "LINK_FIRST"
    SUMMARY_FIRST = "SUMMARY_FIRST"


def response_order_from_str(value: str) -> ResponseOrder:
    if value == "LINK_FIRST":
        return ResponseOrder.LINKS_FIRST
    else:
        return ResponseOrder.SUMMARY_FIRST
