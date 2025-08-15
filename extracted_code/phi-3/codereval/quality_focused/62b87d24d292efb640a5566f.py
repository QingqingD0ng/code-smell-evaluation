class PieceRenderer:
    def __init__(self, style):
        self.style = style

    def apply_style(self, piece):
        # Assuming 'piece' is a dictionary with necessary content
        styled_piece = piece.copy()
        for key, value in self.style.items():
            # Apply style only if the key exists in the piece
            if key in styled_piece:
                styled_piece[key] = value
        return styled_piece


def render(pieces, style):
    renderer = PieceRenderer(style)
    styled_pieces = [renderer.apply_style(piece) for piece in pieces]
    return styled_pieces