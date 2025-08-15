import json

def generate_base_config(user, etcd_host="localhost", etcd_port=2379):
    config = {
        "user": user,
        "tls": False,
        "authentication": "none",
        "authorization": "role-based",
        "etcd": {
            "host": etcd_host,
            "port": etcd_port
        },
        "docs": "https://docs.example.com",
        "log": "https://logs.example.com"
    }
    return json.dumps(config, indent=4)

# Example usage:
# print(generate_base_config("alice", "192.168.1.1", 2380))