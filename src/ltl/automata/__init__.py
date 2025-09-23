from contextlib import nullcontext

from utils import memory, timeit
from .owl import run_owl
from .rabinizer import run_rabinizer
from .ldba import LDBA, LDBATransition
from .ldba_sequence import LDBASequence


@memory.cache
def ltl2ldba(formula: str, propositions: list[str] = None, simplify_labels=True, print_time=False) -> LDBA:
    """Converts an LTL formula to an LDBA using the rabinizer tool."""
    from ltl.hoa import HOAParser
    with timeit(f'Converting LTL formula "{formula}" to LDBA') if print_time else nullcontext():
        hoa = run_rabinizer(formula)
    return HOAParser(formula, hoa, propositions, simplify_labels=simplify_labels).parse_hoa()


@memory.cache
def ltl2nba(formula: str, propositions: list[str] = None, simplify_labels=True, print_time=False) -> LDBA:
    """Converts an LTL formula to a NBA using the OWL tool."""
    from ltl.hoa import HOAParser
    with timeit(f'Converting LTL formula "{formula}" to NBA') if print_time else nullcontext():
        hoa = run_owl(formula)
    return HOAParser(formula, hoa, propositions, simplify_labels=simplify_labels).parse_hoa()


__all__ = ['LDBASequence', 'LDBA', 'LDBATransition', 'ltl2ldba', 'ltl2nba']
