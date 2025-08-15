)', re.DOTALL)
    text_without_code = re.sub(code_block_pattern, '', text)
    tag_pattern = re.compile(r'<(/?)(\w+)>')

    def replacer(match):
        tag = match.group(2)
        return replacer(tag) if replacer else tag

    for tag in tag_pattern.findall(text_without_code):
        if tag.startswith('/'):
            continue
        tags.add(replacer(tag))
        text_without_code = tag_pattern.sub(replacer(tag), text_without_code, 1)

    return tags, text_without_code