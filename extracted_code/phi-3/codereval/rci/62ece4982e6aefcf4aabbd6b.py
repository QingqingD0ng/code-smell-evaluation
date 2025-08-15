def subclasses(cls):
    seen = set()
    queue = [cls]
    result = []

    while queue:
        current_cls = queue.pop(0)
        if current_cls not in seen:
            seen.add(current_cls)
            result.append(current_cls)
            queue.extend(current_cls.__subclasses__())

    return result