def get_versions():
    try:
        import pkg_resources
        versions = {dist.project_name: dist.version for dist in pkg_resources.working_set}
        return versions
    except ImportError:
        return {'default': '1.0'}