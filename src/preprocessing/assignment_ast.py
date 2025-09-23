from abc import ABC


class ASTNode(ABC):
    pass


class NullNode(ASTNode):
    """No assignment (i.e. no transition needs to be avoided)."""
    pass


class EmptyNode(ASTNode):
    """The empty assignments (every proposition is false)."""
    pass


class EpsilonNode(ASTNode):
    """An epsilon transition."""
    pass


class PropositionNode(ASTNode):
    def __init__(self, proposition: str):
        self.proposition = proposition


class AndNode(ASTNode):
    def __init__(self, children):
        self.children = children


class OrNode(ASTNode):
    def __init__(self, children):
        self.children = children
