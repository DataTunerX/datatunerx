
# DataTunerX Comprehensive Deployment Guide

This guide provides detailed instructions for deploying DataTunerX in both online and offline environments. Ensure all prerequisites are met before proceeding with the deployment.

## Prerequisites

Before starting, ensure your system meets the following requirements:

- Kubernetes v1.19+.
- An S3-compatible object storage like Minio for storing large datasets and models.
- A container image registry such as Harbor for securely storing and managing container images.
- Helm, the Kubernetes package manager, for deploying and managing applications.

## Deployment Artifacts

Required artifacts:

- `dtx-ctl` DataTunerX deployment tool.
- `images-ai.tar`: Mandatory large model offline image package.
- `images.tar`: Optional business component offline image package.

## Online Deployment

### 1. Download the `dtx-ctl` Tool

```bash
wget https://github.com/DataTunerX/dtx-ctl/releases/download/v0.1.0/dtx-ctl.tar.gz
```

### 2. Download Base AI Images

```bash
# Placeholder for the actual command to download the base AI images
wget [Your-Base-AI-Image-Package-Download-Link]
```

### 3. Import Base AI Images

Transfer the `images-ai.tar` package to a node with a GPU:

```bash
scp images-ai.tar user@ip:/path/
```

For Docker:

```bash
docker load -i /path/images-ai.tar
```

For Containerd:

```bash
ctr -n k8s.io images import /path/images-ai.tar
```

### 4. Deploy DataTunerX

Deploy with default settings:

```bash
dtx-ctl install
```

Or, with custom settings:

```bash
dtx-ctl install <name> -n <namespace> --set [Your-Custom-Settings]
```

Or, using a configuration file:

```bash
dtx-ctl install <name> -f /path/to/your/config.yaml
```

## Offline Deployment

Follow the online deployment steps for downloading the `dtx-ctl` tool and base images. Additionally, handle the business component images as follows:

### 3. Unzip and Import Business Image Package

```bash
tar -zxcf images.tar -C /path/to/unzip
cd /path/to/unzip/images
```

For Docker:

```bash
docker load -i /path/to/image.tar
```

For Containerd:

```bash
ctr -n k8s.io images import /path/to/image.tar
```

### 4. Modify Image Tags and Push to Your Image Repository

```bash
docker tag source_image:tag target_repository/target_image:tag
docker push target_repository/target_image:tag
```

### 5. Deploy DataTunerX

Deploy using custom settings to configure your image repository:

```bash
dtx-ctl install <name> -n <namespace> --registry=your_registry --repository=your_repository
```

Or, using a configuration file:

```bash
dtx-ctl install <name> -f /path/to/your/config.yaml
```

## Command-Line Command List

Commands to interact with `dtx-ctl`, including flags and subcommands for installation and management:

```bash
# General usage
dtx-ctl [command]

# Available Commands
completion  Generate the autocompletion script for the specified shell
help        Help about any command
install     Install DataTunerX on Kubernetes
uninstall   Uninstall DataTunerX from Kubernetes

# Flags for installation
--chart-directory string     Helm chart directory
--dry-run                    Simulate an install
--image-file-dir string      Specify an image file directory
--image-pull-policy string   Image pull policy
--image-pull-secret string   Image pull secret
--registry string            Container registry
--repository string          Container repository
--set stringArray            Set helm values
--set-file stringArray       Set helm values from files
--set-string stringArray     Set helm STRING values
-f, --values strings         Specify helm values in a YAML file
--version string             Chart version
--wait                       Wait for installation completion
--wait-duration duration     Maximum time to wait for resource readiness
```

Please replace placeholders with actual values and download links as required.
