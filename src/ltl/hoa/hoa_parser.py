from typing import Optional
import re

from ltl.automata import LDBA


class HOAParser:
    """
    A parser for LDBAs given in the HOA format. Handles epsilon transitions as output by rabinizer.
    """

    def __init__(self, formula: str, hoa_text: str, propositions: Optional[set[str]] = None, simplify_labels=True):
        self.formula = formula
        self.lines = hoa_text.split('\n')
        self.line_number = 0
        self.ldba = None
        self.propositions: Optional[set[str]] = propositions
        self.propositions_in_hoa: Optional[list[str]] = None
        self.simplify_labels = simplify_labels

    def parse_hoa(self) -> LDBA:
        if self.ldba is not None:
            return self.ldba
        self.parse_propositions()
        self.ldba = LDBA(self.propositions, formula=self.formula, simplify_labels=self.simplify_labels)
        self.parse_header()
        self.parse_body()
        return self.ldba

    def parse_propositions(self):
        self.propositions_in_hoa = self.find_and_parse_ap_line()
        if self.propositions is None:
            self.propositions = set(self.propositions_in_hoa)
        else:
            if not set(self.propositions_in_hoa).issubset(self.propositions):
                raise ValueError(
                    'Error parsing HOA. Found propositions in header that do not match given propositions.')

    def find_and_parse_ap_line(self) -> list[str]:
        for num, line in enumerate(self.lines):
            if line.startswith('AP:'):
                return self.parse_ap_line(line.split(':')[1].strip(), num)
        raise ValueError('Error parsing HOA. Missing required header field `AP`.')

    @staticmethod
    def parse_ap_line(value: str, line_number: int) -> list[str]:
        parts = value.split(' ')
        num_props = parts[0]
        props = parts[1:]
        if int(num_props) != len(props):
            raise ValueError(f'Error parsing HOA at line {line_number}. Expected {num_props} propositions.')
        return [p.replace('"', '') for p in props]

    def parse_header(self):
        self.expect_line('HOA: v1')
        found_start = False
        while self.peek(error_msg='Expecting "--BODY--".') != '--BODY--':
            name, value = self.parse_header_line()
            match name:
                case 'Start':
                    self.ldba.add_state(int(value), initial=True)
                    found_start = True
                case 'acc-name':
                    self.expect('Buchi', value)
                case 'Acceptance':
                    self.expect('1 Inf(0)', value)
                case _:
                    continue
        if not found_start:
            raise ValueError(
                f'Error parsing HOA at line {self.line_number}. Missing required header field `Start`.')

    def parse_header_line(self) -> tuple[str, str]:
        line = self.consume(error_msg="Expecting header line.")
        if ':' not in line:
            raise ValueError(f'Error parsing HOA at line {self.line_number}. Expected a header line.')
        name, value = line.split(':')
        return name.strip(), value.strip()

    def parse_body(self):
        self.expect_line('--BODY--')
        while self.peek(error_msg='Expecting "--END--".') != '--END--':
            self.parse_state()
        self.expect_line('--END--')

    def parse_state(self):
        state_line = self.consume(error_msg='Expecting state line.')
        if not state_line.startswith('State: '):
            raise ValueError(f'Error parsing HOA at line {self.line_number}. Expected a state line.')
        state = int(state_line.split(' ')[1])
        self.ldba.add_state(state)
        while self.peek().startswith('[') or self.peek().isdigit():
            self.parse_transition(state)

    def parse_transition(self, source: int):
        line = self.consume(error_msg='Expecting transition line.')
        label, line = self.parse_label(line)
        parts = line.split(' ')
        target = int(parts[0])
        self.ldba.add_state(target)
        accepting = False
        if len(parts) > 1:
            self.expect('{0}', parts[1])
            accepting = True
        self.ldba.add_transition(source, target, label, accepting)

    def parse_label(self, line: str) -> tuple[Optional[str], str]:
        label = None
        if line.startswith('['):
            parts = line.split(']')
            label = parts[0][1:].strip()
            label = self.replace_numeric_propositions(label)
            line = parts[1].strip()
        return label, line

    def replace_numeric_propositions(self, label: str) -> str:
        assert self.propositions_in_hoa is not None
        regexp = r'(\d+)'
        return re.sub(regexp, lambda m: self.propositions_in_hoa[int(m.group(0))], label)

    def peek(self, error_msg: Optional[str] = None) -> str:
        if self.line_number >= len(self.lines):
            raise ValueError(f'Error parsing HOA. Reached end of input.{"" if error_msg is None else f" {error_msg}"}')
        return self.lines[self.line_number]

    def consume(self, error_msg: Optional[str] = None) -> str:
        line = self.peek(error_msg)
        self.line_number += 1
        return line

    def expect_line(self, expected: str):
        if self.peek() != expected:
            raise ValueError(f'Error parsing HOA at line {self.line_number}. Expected: {expected}.')
        self.line_number += 1

    def expect(self, expected: any, actual: any):
        if expected != actual:
            raise ValueError(f'Error parsing HOA at line {self.line_number}. Expected: {expected}.')
