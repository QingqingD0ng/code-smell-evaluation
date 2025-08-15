def get_versions():

    try:

        import pkg_resources

        return pkg_resources.get_distribution("my_package").version

    except Exception:

        return "1.0.0"  # Default version