config = {
    "tls": False,
    "authentication": "basic",
    "authorization": "token",
    "etcd": {
        "host": etcd_host,
        "port": etcd_port
    },
    "docs": True,
    "log": {
        "level": "info"
    }
}

# Usage
base_config_dict = base_config("user")
print(base_config_dict)