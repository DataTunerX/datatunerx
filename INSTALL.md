## Prerequisites:

- Kubernetes v1.19+
- Minio (or other object storage supporting S3 protocol)
- Harbor (or other image repository)
- Helm

## Deployment Artifacts:

- dtx-ctl DataTunerX deployment tool
- images-ai.tar Large model offline image package (mandatory)
- images.tar Business component offline image package (optional)

## Online Deployment:

1. Download the dtx-ctl command-line tool:

```bash
wget https://github.com/DataTunerX/dtx-ctl/releases/download/v0.1/dtx-ctl.tar.gz
```

2. Import base images:

   Transfer the large model image from the image package to a node with a GPU:

   ```bash
   scp images-ai.tar user@ip:/path/
   ```

   - For Docker container runtime:

   ```bash
   docker load -i images-ai.tar
   ```

   - For Containerd container runtime:

   ```bash
   ctr -n k8s.io images import images-ai.tar
   ```

3. Deploy DataTunerX using dtx-ctl:

```bash
## Install datatunerx with default settings
dtx-ctl install 
## Install datatunerx with custom settings
dtx-ctl install <name> -n <namespace> --set 
## Install datatunerx using a configuration file
dtx-ctl install <name> -f config.yaml
```

## Offline Deployment:

1. Download the dtx-ctl command-line tool:

```bash
wget https://github.com/DataTunerX/dtx-ctl/releases/download/v0.1/dtx-ctl.tar.gz
```

2. Import base images:

   Transfer the large model image from the image package to a node with a GPU:

   ```bash
   scp images-ai.tar user@ip:/path/
   ```

   - For Docker container runtime:

   ```bash
   docker load -i images-ai.tar
   ```

   - For Containerd container runtime:

   ```bash
   ctr -n k8s.io images import images-ai.tar
   ```

   3. Unzip the business image package on the current machine node:

   ```bash
   tar -zxcf images.tar
   ## Enter the extracted image folder
   cd images
   ## Import all images in the current directory
   ## For Docker container runtime:
   docker load -i current_folder_directory
   ## For Containerd container runtime:
   ctr -n k8s.io images import images.tar
   ```

   4. If you have an image repository, modify the image tag and push it to your image repository:

   ```bash
   docker tag registry/repository/image:tag yourregistry/yourrepository/image:tag
   docker push yourregistry/yourrepository/image:tag
   ```

3. Deploy DataTunerX using dtx-ctl:

```bash
## Install datatunerx with default settings
dtx-ctl install 
## Install datatunerx with custom settings, configure your image repository address and repository
dtx-ctl install <name> -n <namespace> --registry=registry.cn-hangzhou.aliyuncs.com --repository=datatunerx
## Install datatunerx using a configuration file
dtx-ctl install <name> -f config.yaml
```

## Command Line Command List:

- Parent command:

```bash
Usage:
  dtx-ctl [flags]
  dtx-ctl [command]

Available Commands:
  completion  Generate the autocompletion script for the specified shell
  help        Help about any command
  install     Install datatunerx by helm on kubernetes
  uninstall   unInstall datatunerx by helm on kubernetes
```

- Subcommands

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

Global Flags:
      --context string     Kubernetes configuration context
  -n, --namespace string   Namespace datatunerx is running in (default "datatunerx-dev")