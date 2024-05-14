import logging

from multilingual_text_parser.processors.ru.normalizer_rules import *
from multilingual_text_parser.processors.ru.normalizer_utils import BaseNormalizer

__all__ = ["RuleBasedNormalizerRU"]

LOGGER = logging.getLogger("root")


class RuleBasedNormalizerRU(BaseNormalizer):
    def __init__(self):
        super().__init__()
        # Правила для SSML
        self._rules_ssml = [
            CharactersSSML(self._morph, self._num2words),
        ]
        # Правила в начале
        self._rules_begin = [
            AddressRules(self._morph, self._num2words),
            ReductionInDocsRules(self._morph, self._num2words),
            NumAdjRules(self._morph, self._num2words),
            PhoneRules(self._morph, self._num2words),
        ]
        self._rules_tagged = {
            "time": TimeRulesTag(self._morph, self._num2words),
            "date": DateRulesTag(self._morph, self._num2words),
            "roman": RomanRulesTag(self._morph, self._num2words),
            "fraction": FractionRulesTag(self._morph, self._num2words),
            "ordinal": OrdinalRulesTag(self._morph, self._num2words),
            "digit": DigitRulesTag(self._morph, self._num2words),
        }
        # Остальные правила
        self._rules = [
            # AddressRules(self._morph, self._num2words),
            # FractionRules(self._morph, self._num2words),
            # Num_AdjRules(self._morph, self._num2words),
            # DataPeriodRules(self._morph, self._num2words),
            # YearPeriodRules(self._morph, self._num2words),
            # OneAndHalfRules(self._morph, self._num2words),
            # ReductionInDocsRules(self._morph, self._num2words),
            # YearRules(self._morph, self._num2words),
            # DateRules(self._morph, self._num2words),
            TimeRules(self._morph, self._num2words),
            # ScoreRules(self._morph, self._num2words),
            # PhoneRules(self._morph, self._num2words),
            # DigitRules(self._morph, self._num2words),
            RomanRules(self._morph, self._num2words),
            DigitRules(self._morph, self._num2words),
            Translit(self._morph, self._num2words),
        ]
