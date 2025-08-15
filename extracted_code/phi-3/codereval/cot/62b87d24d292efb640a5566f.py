def render(pieces, style):

    return [apply_style(piece, style) for piece in pieces]


def apply_style(piece, style):

    styled_piece = f"{piece} in style {style}"

    return styled_piece


pieces = ["Piece1", "Piece2", "Piece3"]

style = "bold"

styled_pieces = render(pieces, style)

print(styled_pieces)