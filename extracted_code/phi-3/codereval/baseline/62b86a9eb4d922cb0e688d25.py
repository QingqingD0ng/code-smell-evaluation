import re

def _get_resource_name_regex():
    return {
        'api_gateway': re.compile(r'^api-gateway-.*$'),
        'aws_ec2': re.compile(r'^aws-ec2-.*$'),
        'azure_vm': re.compile(r'^azure-vm-.*$'),
        'gcp_compute': re.compile(r'^gcp-compute-.*$'),
        'kubernetes_deployment': re.compile(r'^kubernetes-deployment-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'),
        'helm_chart': re.compile(r'^helm-chart-.*v\d+\.\d+\.\d+$'),
    }