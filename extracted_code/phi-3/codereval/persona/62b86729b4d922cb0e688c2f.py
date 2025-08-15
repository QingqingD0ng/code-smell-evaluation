base_config = {
    "tls": False,
    "authentication": "basic",
    "authorization": "token",
    "etcd": {
        "host": etcd_host,
        "port": etcd_port
    },
    "docs": True,
    "log": True
}

def base_config(user, etcd_host="localhost", etcd_port=2379):
    return {
        "user": user,
        **base_config(etcd_host, etcd_port)
    }