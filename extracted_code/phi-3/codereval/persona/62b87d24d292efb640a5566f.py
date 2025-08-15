def render(pieces, style):

    stylized_pieces = []

    for piece in pieces:

        if style == 'uppercase':

            stylized_pieces.append(piece.upper())

        elif style == 'lowercase':

            stylized_pieces.append(piece.lower())

        elif style =='reverse':

            stylized_pieces.append(piece[::-1])

        else:

            raise ValueError(f"Unsupported style: {style}")

    return stylized_pieces