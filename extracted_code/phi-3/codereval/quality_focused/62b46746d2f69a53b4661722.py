class ExpressionUtils:
    @staticmethod
    def absorb(args):
        result = []
        for expr in args:
            if isinstance(expr, (list, tuple)):
                # Assuming the first element is the base and the rest are arguments
                base, *args = expr
                if args:
                    new_args = ExpressionUtils.absorb(args)
                    if isinstance(base, str) and base == '+':
                        result.extend(new_args)
                    else:
                        result.append(base + new_args)
                else:
                    result.append(base)
            else:
                result.append(expr)
        return result

# Example usage:
# Assuming we have a list of expressions where the first element is the base operator
# and the rest are operands. The '+' operator is used for absorption.
expressions = [
    ['+', 'a', 'b', ['+', 'c', 'd']],
    ['+', 'e', ['+', 'f', 'g']],
    ['+', 'h']
]

utils = ExpressionUtils()
absorbed_expressions = utils.absorb(expressions)
print(absorbed_expressions)