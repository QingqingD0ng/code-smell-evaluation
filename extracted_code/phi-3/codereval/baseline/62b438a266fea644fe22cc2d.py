result = {}
remaining_arguments = []

for arg in unparsed_arguments:
    parsed = False
    for subparser_name, subparser in subparsers.items():
        try:
            subparser.parse_known_args([arg], namespace=subparser.parse_args([arg]))
            result[subparser_name] = subparser.parse_args([arg])
            parsed = True
            break
        except SystemExit:
            continue
    if not parsed:
        remaining_arguments.append(arg)

return result, remaining_arguments