def paging(response, max_results):
    for i in range(0, len(response), max_results):
        yield response[i:i + max_results]