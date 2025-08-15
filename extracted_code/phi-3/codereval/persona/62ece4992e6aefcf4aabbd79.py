import re


def make_find_paths(find_paths):
	patterns = []

	for path in find_paths:
		# Check if the path is already a glob pattern
		if re.match(r'^[\*?]', path):
			patterns.append(path)
		else:
			# Convert the path to a glob pattern
			patterns.append(re.sub(r'([\/\.\-])', r'\\\1', path).replace('*', '*').replace('?', '?'))

	return tuple(patterns)