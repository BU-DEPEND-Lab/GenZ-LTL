import functools
from abc import ABC, abstractmethod
from typing import Optional
from ltl.logic.boolean_lexer import Lexer, Token, TokenType


class Parser:
    def __init__(self, expression: str):
        lexer = Lexer(expression)
        self.tokens: list[Token] = lexer.lex()
        self.pos = 0
        self.current_token: Optional[Token] = self.tokens[self.pos] if self.tokens else None

    def parse(self) -> 'Node':
        result = self.parse_expression()
        if self.current_token is not None:
            raise SyntaxError(f"Unexpected token at the end: {self.current_token}")
        return result

    def parse_expression(self) -> 'Node':
        node = self.parse_implication()
        while self.match(TokenType.OR):
            right = self.parse_implication()
            node = OrNode(node, right)
        return node

    def parse_implication(self) -> 'Node':
        node = self.parse_or()
        while self.match(TokenType.IMPLIES):
            right = self.parse_or()
            node = ImplicationNode(node, right)
        return node

    def parse_or(self) -> 'Node':
        node = self.parse_and()
        while self.match(TokenType.OR):
            right = self.parse_and()
            node = OrNode(node, right)
        return node

    def parse_and(self) -> 'Node':
        node = self.parse_primary()
        while self.match(TokenType.AND):
            right = self.parse_primary()
            node = AndNode(node, right)
        return node

    def parse_primary(self) -> 'Node':
        if self.match(TokenType.NOT):
            return NotNode(self.parse_primary())
        elif self.match(TokenType.LPAREN):
            node = self.parse_expression()
            if not self.match(TokenType.RPAREN):
                raise SyntaxError("Expected ')'")
            return node
        else:
            return self.parse_variable()

    def parse_variable(self) -> 'Node':
        if self.current_token and self.current_token.type == TokenType.VAR:
            node = VarNode(self.current_token.value)
            self.next_token()
            return node
        else:
            raise SyntaxError(f"Unexpected token: {self.current_token}")

    def match(self, token_type: TokenType) -> bool:
        if self.current_token and self.current_token.type == token_type:
            self.next_token()
            return True
        return False

    def next_token(self) -> None:
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current_token = self.tokens[self.pos]
        else:
            self.current_token = None


class Node(ABC):
    @abstractmethod
    def eval(self, context: dict[str, bool]) -> bool:
        pass


class AndNode(Node):
    def __init__(self, left: Node, right: Node):
        self.left = left
        self.right = right

    def eval(self, context: dict[str, bool]) -> bool:
        return self.left.eval(context) and self.right.eval(context)


class OrNode(Node):
    def __init__(self, left: Node, right: Node):
        self.left = left
        self.right = right

    def eval(self, context: dict[str, bool]) -> bool:
        return self.left.eval(context) or self.right.eval(context)


class NotNode(Node):
    def __init__(self, operand: Node):
        self.operand = operand

    def eval(self, context: dict[str, bool]) -> bool:
        return not self.operand.eval(context)


class VarNode(Node):
    def __init__(self, name: str):
        self.name = name

    def eval(self, context: dict[str, bool]) -> bool:
        return context[self.name]


class ImplicationNode(Node):
    def __init__(self, left: Node, right: Node):
        self.left = left
        self.right = right

    def eval(self, context: dict[str, bool]) -> bool:
        return not self.left.eval(context) or self.right.eval(context)


@functools.lru_cache(maxsize=500_000)
def parse(expression: str) -> Node:
    return Parser(expression).parse()


if __name__ == '__main__':
    expression = 'a | b | c'
    parser = Parser(expression)
    ast = parser.parse()

    # Define the context with variable values
    context = {'a': True, 'b': False, 'c': True}
    result = ast.eval(context)
    print(result)  # Output: True
