
# DataTunerX Comprehensive Deployment Guide

This guide provides detailed instructions for deploying DataTunerX in both online and offline environments. Ensure all prerequisites are met before proceeding with the deployment.

## Prerequisites

Before starting, ensure your system meets the following requirements:

- **Kubernetes v1.19+**: The container orchestration system for automating software deployment, scaling, and management.
- **Minio** or another S3-compatible object storage: For storing large datasets and models.
- **Harbor** or another container image registry: For securely storing and managing container images.
- **Helm**: The Kubernetes package manager for deploying and managing applications.

## Deployment Artifacts

Required artifacts:

- `dtx-ctl` DataTunerX deployment tool.
- `images-ai.tar`: Optional llm offline image package. (The image size is 47.1GB)
- `images.tar`: Optional business component offline image package.

## Online Deployment

### 1. Download the `dtx-ctl` Tool

```bash
wget https://github.com/DataTunerX/dtx-ctl/releases/download/v0.1.0/dtx-ctl.tar.gz
```

### 2. Deploy DataTunerX

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

### 1. Download the `dtx-ctl` Tool

```bash
wget https://github.com/DataTunerX/dtx-ctl/releases/download/v0.1.0/dtx-ctl.tar.gz
```

### 2. Download Base Images

```bash
# Placeholder for the actual command to download the base AI images, currently the link is valid for 24 hours, if you need to apply for the download package please mention issuer
wget https://public-download.daocloud.io/datatunerx/v0.1.0/images?e=1708664238&token=MHV7x1flrG19kzrdBNfPPO7JpBjTr__AMGzOtlq1:sZrIxT02pubO4BhPunS3sky3Fss=
```

### 3. Download Base AI Images

```bash
# Placeholder for the actual command to download the base AI images, currently the link is valid for 24 hours, if you need to apply for the download package please mention issuer
wget https://public-download.daocloud.io/datatunerx/v0.1.0/images-ai?e=1708594433&token=MHV7x1flrG19kzrdBNfPPO7JpBjTr__AMGzOtlq1:DySesLobN0I7NeCBcYuZ74P8osA=
```

### 4. Unzip and Import Business Image Package

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

### 5. Modify Image Tags and Push to Your Image Repository

```bash
docker tag source_image:tag target_repository/target_image:tag
docker push target_repository/target_image:tag
```

### 6. Deploy DataTunerX

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
