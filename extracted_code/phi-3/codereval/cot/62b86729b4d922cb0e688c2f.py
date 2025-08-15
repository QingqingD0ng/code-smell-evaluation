class Config:
    def __init__(self, user):
        self.user = user
        self.etcd_host = "localhost"
        self.etcd_port = 2379
        self.tls = False
        self.authentication = "basic"
        self.authorization = "token"
        self.docs = True
        self.log = True

    def update_etcd(self, etcd_host=None, etcd_port=None):
        if etcd_host is not None:
            self.etcd_host = etcd_host
        if etcd_port is not None:
            self.etcd_port = etcd_port

    def toggle_tls(self):
        self.tls = not self.tls

    def toggle_docs(self):
        self.docs = not self.docs

    def toggle_log(self):
        self.log = not self.log

    def __str__(self):
        return (
            f"User: {self.user}\n"
            f"ETCD Host: {self.etcd_host}\n"
            f"ETCD Port: {self.etcd_port}\n"
            f"TLS: {self.tls}\n"
            f"Authentication: {self.authentication}\n"
            f"Authorization: {self.authorization}\n"
            f"Docs: {self.docs}\n"
            f"Log: {self.log}\n"
        )

# Example usage:
config = Config(user="admin")
print(config)

config.update_etcd(etcd_host="192.168.1.1", etcd_port=2379)
print(config)

config.toggle_tls()
print(config)

config.toggle_docs()
print(config)

config.toggle_log()
print(config)