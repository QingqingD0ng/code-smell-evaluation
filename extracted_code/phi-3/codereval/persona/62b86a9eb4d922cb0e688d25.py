def _get_resource_name_regex():

    return {

        'Cluster': r'^cluster-[a-zA-Z0-9\-]+$',

        'ClusterRole': r'^clusterrole-[a-zA-Z0-9\-]+$',

        'ClusterRoleBinding': r'^clusterrolebinding-[a-zA-Z0-9\-]+$',

        'Config': r'^config-[a-zA-Z0-9\-]+$',

        'Node': r'^node-[a-zA-Z0-9\-]+$',

        'NodeRole': r'^noderole-[a-zA-Z0-9\-]+$',

        'NodeRoleBinding': r'^noderolebinding-[a-zA-Z0-9\-]+$',

        'Project': r'^project-[a-zA-Z0-9\-]+$',

        'ProjectRole': r'^projectrole-[a-zA-Z0-9\-]+$',

        'ProjectRoleBinding': r'^projectrolebinding-[a-zA-Z0-9\-]+$',

        'StorageClass': r'^storageclass-[a-zA-Z0-9\-]+$',

        'StorageClassSpec': r'^storageclassspec-[a-zA-Z0-9\-]+$',

        'Snapshots': r'^snapshot-[a-zA-Z0-9\-]+$',

        'SnapshotsList': r'^snapshotslist-[a-zA-Z0-9\-]+$',

        'Snapshot': r'^snapshot-[a-zA-Z0-9\-]+$',

        'SnapshotList': r'^snapshotslist-[a-zA-Z0-9\-]+$',

        'SnapshotPolicy': r'^snapshotpolicy-[a-zA-Z0-9\-]+$',

        'SnapshotPolicySpec': r'^snapshotpolicyspec-[a