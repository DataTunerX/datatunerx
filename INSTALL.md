
# DataTunerX Deployment Guide

## Prerequisites

Ensure you have the following prerequisites installed and configured:

- **Kubernetes v1.19+**: The container orchestration system for automating software deployment, scaling, and management.
- **Minio** or other S3-compatible object storage: For storing large datasets, models, and other blobs.
- **Harbor** or another container image registry: To securely store and manage container images.
- **Helm**: A package manager for Kubernetes, facilitating the deployment and management of applications.

## Deployment Artifacts

You will need the following artifacts for deployment:

- **dtx-ctl DataTunerX deployment tool**: A CLI tool to streamline the deployment of DataTunerX on Kubernetes.
- **images-ai.tar**: This is a mandatory offline image package containing large AI models. The download link is provided separately.
- **images.tar**: This is an optional business component offline image package, available via a separate download link.

## Online Deployment Guide

Follow these steps for an online deployment of DataTunerX:

1. **Download the `dtx-ctl` Command-Line Tool**:
```bash
wget https://github.com/DataTunerX/dtx-ctl/releases/download/v0.1.0/dtx-ctl.tar.gz
```

2. **Download Base AI Images**:
The command to download the base AI images is missing. Ensure you have the correct URL to download this package.

3. **Import Base Images**:
- Transfer the large model image package to a node with a GPU using `scp`.
- Load the images into your container runtime:
- For Docker: `docker load -i images-ai.tar`
- For Containerd: `ctr -n k8s.io images import images-ai.tar`

4. **Deploy DataTunerX Using `dtx-ctl`**:
Deploy DataTunerX with default settings, custom settings, or using a configuration file.

## Offline Deployment Guide

For an offline deployment, the steps are similar, with additional steps for handling the business component images:

1. **Download the `dtx-ctl` Command-Line Tool** and **Base Images** as in the online deployment guide.

2. **Import Base Images**:
Import both the AI model images and the business component images into your container runtime as described in the online deployment guide.

3. **Unzip and Import Business Image Package**:
- Extract the business image package.
- Load all images in the current directory into your container runtime.

4. **Modify Image Tags and Push to Your Image Repository** (if applicable):
Tag the images with your repository's address and push them.

5. **Deploy DataTunerX Using `dtx-ctl`**:
Deploy with default settings, configure your image repository, or use a configuration file.

## Command-Line Command List

- **Parent Command Usage**:

```bash
completion  Generate the autocompletion script for the specified shell
help        Help about any command
install     Install datatunerx by helm on kubernetes
uninstall   unInstall datatunerx by helm on kubernetes
```

- **Subcommands and Flags**:

```bash
--chart-directory string     Helm chart directory
--dry-run                    Simulate an install
--image-file-dir string      Specify an image file dir for the chart version to use. For example, --image-file-dir=/tmp
--image-file-dir=/tmp
--image-pull-policy string   Specify an image pull policy for the chart version to use. For example, --image-pull-policy=Always
--image-pull-secret string   Specify an image pull secret for the chart version to use. For example, --image-pull-secret=datatunerx
--push                       Specify a push for the chart version to use. For example, --push=true
--registry string            Specify a registry for the chart version to use. For example, --registry=registry.cn-hangzhou.aliyuncs.com
--repository string          Specify a repository for the chart version to use. For example, --repository=datatunerx
--set stringArray            Set helm values on the command line (can specify multiple or separate values with commas: key1=val1,key2=val2)
--set-file stringArray       Set helm values from respective files specified via the command line (can specify multiple or separate values with commas: key1=path1,key2=path2)
--set-string stringArray     Set helm STRING values on the command line (can specify multiple or separate values with commas: key1=val1,key2=val2)
-f, --values strings             Specify helm values in a YAML file or a URL (can specify multiple)
--version string             Specify a version constraint for the chart version to use. For example, --version=0.0.1 (default "0.0.1")
--wait                       Wait for installation to have completed
--wait-duration duration     Maximum time to wait for resources to be ready (default 5m0s)
```
