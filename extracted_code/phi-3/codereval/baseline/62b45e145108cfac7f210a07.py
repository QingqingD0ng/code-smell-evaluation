self.spec_version = inventory.get('type', '').split('-')[0] if extract_spec_version and 'type' in inventory else self.spec_version

	if not self.spec_version:
		self.spec_version = 'unknown'

	if '-' not in self.spec_version:
		raise ValueError("Invalid specification version format.")

	version_major, version_minor = self.spec_version.split('-')
	version_major, version_minor = int(version_major), int(version_minor)

	if version_major < self.min_version_major or (version_major == self.min_version_major and version_minor < self.min_version_minor):
		raise ValueError("Specification version is too old.")

	if version_major > self.max_version_major:
		raise ValueError("Specification version is too new.")

	self.is_valid = True