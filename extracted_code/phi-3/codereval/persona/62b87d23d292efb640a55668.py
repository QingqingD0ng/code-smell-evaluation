import versioneer

def get_config():
    config = versioneer.Config()
    config.root = '.'
    config.versionfile_source ='src/VERSION'
    config.versionfile_build = None
    config.tag_builds = True
    config.writes_to = None
    config.write_to_template = '{{root}}/{{tag}}/{{version}}'
    config.dev_status ='stable'
    config.git_rev = 'HEAD'
    config.dev_url = 'https://github.com/your-username/your-repo'
    config.project_url = 'https://github.com/your-username/your-repo'
    config.release_classifier = 'Development Status :: 4 - Beta'
    return config