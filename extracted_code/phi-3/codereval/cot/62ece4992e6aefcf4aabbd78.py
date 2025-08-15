from ipaddress import ip_address, ip_network

LOCALHOST_NAMES = [
    "localhost",
    "127.0.0.1",
    "::1",
    "fe80::1",
    "home.local",
    "localhost.localdomain",
    "local",
    "localdomain"
]

LOCAL_IPV4_RANGE = ip_network("127.0.0.0/8")
LOCAL_IPV6_RANGE = ip_network("::1/128")

def is_local(host):
    try:
        host_ip = ip_address(host)
    except ValueError:
        return False

    if host_ip in LOCAL_IPV4_RANGE:
        return True

    if host_ip in LOCAL_IPV6_RANGE:
        return True

    for name in LOCALHOST_NAMES:
        if host == name:
            return True

    return False