from enum import StrEnum


class Market(StrEnum):
    HK = "HK"
    CN = "CN"
    SG = "SG"
    JP = "JP"
    KR = "KR"
    AU = "AU"
    TW = "TW"


# Markets that trade two distinct sessions - only these accept session="am"/"pm" on a PricePoint
DUAL_SESSION_MARKETS = frozenset({Market.HK, Market.CN, Market.SG})
