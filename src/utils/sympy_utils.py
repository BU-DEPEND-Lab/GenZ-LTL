import functools

import sympy


@functools.cache
def to_sympy(label: str) -> sympy.logic.boolalg.Boolean:
    if label == 't':
        return sympy.logic.boolalg.true
    return sympy.sympify(label.replace('!', '~'))


def sympy_to_str(formula: sympy.logic.boolalg.Boolean) -> str:
    result = str(formula).replace('~', '!')
    if result == 'True':
        return 't'
    return result


@functools.cache
def simplify(formula: sympy.logic.boolalg.Boolean) -> sympy.logic.boolalg.Boolean:
    return formula.simplify()
