def validate_as_prior_version(self, prior):
    if not isinstance(prior, InventoryValidator):
        return Error()

    if self.version < prior.version:
        return Error()

    for field, value in self.fields.items():
        if field in prior.fields and value!= prior.fields[field]:
            return Error()

    return None  # No error, prior is a valid prior version