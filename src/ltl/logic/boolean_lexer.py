from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TokenType(Enum):
    NOT = '!'
    AND = '&'
    OR = '|'
    IMPLIES = '=>'
    LPAREN = '('
    RPAREN = ')'
    VAR = 'VAR'


@dataclass(eq=True)
class Token:
    type: TokenType
    value: str

    def __repr__(self):
        return f"Token({self.type}, {repr(self.value)})"


class Lexer:
    def __init__(self, formula: str):
        self.formula = formula
        self.pos = 0
        self.length = len(formula)

    def lex(self) -> list[Token]:
        tokens = []
        while self.pos < self.length:
            current_char = self.formula[self.pos]

            if current_char.isspace():
                self.pos += 1
                continue

            if current_char.isalpha() or current_char == '_':
                tokens.append(self.tokenize_variable())
            elif current_char == '!':
                tokens.append(Token(TokenType.NOT, '!'))
                self.pos += 1
            elif current_char == '&':
                tokens.append(Token(TokenType.AND, '&'))
                self.pos += 1
            elif current_char == '|':
                tokens.append(Token(TokenType.OR, '|'))
                self.pos += 1
            elif current_char == '=':
                if self.peek() == '>':
                    tokens.append(Token(TokenType.IMPLIES, '=>'))
                    self.pos += 2
                else:
                    raise SyntaxError(f"Unexpected token: {current_char}")
            elif current_char in '()':
                tokens.append(Token(TokenType.LPAREN if current_char == '(' else TokenType.RPAREN, current_char))
                self.pos += 1
            else:
                raise SyntaxError(f"Unexpected token: {current_char}")

        return tokens

    def tokenize_variable(self) -> Token:
        start_pos = self.pos
        while self.pos < self.length and (self.formula[self.pos].isalnum() or self.formula[self.pos] == '_'):
            self.pos += 1
        value = self.formula[start_pos:self.pos]
        return Token(TokenType.VAR, value)

    def peek(self) -> Optional[str]:
        if self.pos + 1 < self.length:
            return self.formula[self.pos + 1]
        return None


# Usage example
if __name__ == '__main__':
    lexer = Lexer('a & green | !c => d')
    tokens = lexer.lex()
    for token in tokens:
        print(token)
