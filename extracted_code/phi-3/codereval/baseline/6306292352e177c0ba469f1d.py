' and text[i-1]!= '\\':
            in_code_block = not in_code_block
        if not in_code_block:
            if text[i] == '<':
                j = i + 1
                while j < len(text) and text[j]!= '>':
                    j += 1
                tag = text[i+1:j]
                if replacer:
                    result_text.append(replacer(tag))
                    continue
                tags.add(tag)
                i = j
            else:
                result_text.append(text[i])
                i += 1
        else:
            result_text.append(text[i])
            i += 1
    return tags, ''.join(result_text)