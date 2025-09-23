import re

from ltl.automata import LDBA


class HOAWriter:
    def __init__(self, ldba: LDBA):
        self.lines = []
        self.ldba = ldba

    def get_hoa(self) -> str:
        if not self.lines:
            self.write_header()
            self.write_body()
        return '\n'.join(self.lines)

    def write_header(self):
        self.write_line('HOA: v1')
        self.write_line('tool: "deep-ltl"')
        self.write_line(f'Start: {self.ldba.initial_state}')
        self.write_line('acc-name: Buchi')
        self.write_line('Acceptance: 1 Inf(0)')
        self.write_line('properties: trans-acc trans-label')
        props = sorted(self.ldba.propositions) + ['eps']
        self.write_line('AP: {} {}'.format(len(props),
                                           ' '.join([f'"{p}"' for p in props])))

    def write_body(self):
        self.write_line('--BODY--')
        for state in range(self.ldba.num_states):
            self.write_line(f'State: {state}')
            for transition in self.ldba.state_to_transitions[state]:
                line = f'[{self.label_to_index_label(transition.label)}] {transition.target}'
                if transition.accepting:
                    line += ' {0}'
                self.write_line(line)
        self.write_line('--END--')

    def label_to_index_label(self, label: str) -> str:
        if label is None:
            return str(len(self.ldba.propositions))
        regexp = r'([a-z][a-z0-9_]*)'  # matches a single proposition
        return re.sub(regexp, lambda m: str(self.proposition_to_index(m.group(0))), label)

    def proposition_to_index(self, proposition: str) -> int | str:
        if proposition == 't':
            return proposition
        return sorted(self.ldba.propositions).index(proposition)

    def write_line(self, line: str):
        self.lines.append(line)
