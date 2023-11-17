package generate

import (
	"fmt"

	corev1beta1 "github.com/DataTunerX/meta-server/api/core/v1beta1"
	extensionv1beta1 "github.com/DataTunerX/meta-server/api/extension/v1beta1"
	finetunev1beta1 "github.com/DataTunerX/meta-server/api/finetune/v1beta1"
	rayv1 "github.com/ray-project/kuberay/ray-operator/apis/ray/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

func GenerateFinetune(finetuneJob *finetunev1beta1.FinetuneJob) *finetunev1beta1.Finetune {
	if finetuneJob.Spec.FineTune.Name == nil {
		name := fmt.Sprintf("%s-%s", finetuneJob.Name, "finetune")
		finetuneJob.Spec.FineTune.Name = &name
	}
	finetune := &finetunev1beta1.Finetune{
		ObjectMeta: metav1.ObjectMeta{
			Name:      *finetuneJob.Spec.FineTune.Name,
			Namespace: finetuneJob.Namespace,
		},
		Spec: finetunev1beta1.FinetuneSpec{
			Dataset:        finetuneJob.Spec.FineTune.FinetuneSpec.Dataset,
			LLM:            finetuneJob.Spec.FineTune.FinetuneSpec.LLM,
			Hyperparameter: finetuneJob.Spec.FineTune.FinetuneSpec.Hyperparameter,
			Image:          finetuneJob.Spec.FineTune.FinetuneSpec.Image,
			Node:           finetuneJob.Spec.FineTune.FinetuneSpec.Node,
		},
	}
	if finetuneJob.Spec.FineTune.FinetuneSpec.Resource != nil {
		finetune.Spec.Resource = finetuneJob.Spec.FineTune.FinetuneSpec.Resource
	}
	return finetune
}

// todo(tigerK) add build image job
func GenerateBuildImageJob(name, namespace, endpoint, accessKeyId, secretAccessKey,
	bucket, filePath, image, secure, mountPath, registryUrl, repositoryName, username, password, imageName, imageTag string) *batchv1.Job {
	privileged := true
	directory := corev1.HostPathDirectory
	return &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: batchv1.JobSpec{
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:  "imagebuild",
							Image: image,
							Env: []corev1.EnvVar{
								{
									Name:  "S3_ENDPOINT",
									Value: endpoint,
								},
								{Name: "S3_ACCESSKEYID",
									Value: accessKeyId,
								},
								{
									Name:  "S3_SECRETACCESSKEY",
									Value: secretAccessKey,
								},
								{
									Name:  "S3_BUCKET",
									Value: bucket,
								},
								{
									Name:  "S3_FILEPATH",
									Value: filePath,
								},
								{
									Name:  "S3_SECURE",
									Value: secure,
								},
								{
									Name:  "MOUNT_PATH",
									Value: mountPath,
								},
								{
									Name:  "REGISTRY_URL",
									Value: registryUrl,
								},
								{
									Name:  "REPOSITORY_NAME",
									Value: repositoryName,
								},
								{
									Name:  "USERNAME",
									Value: username,
								},
								{
									Name:  "PASSWORD",
									Value: password,
								},
								{
									Name:  "IMAGE_NAME",
									Value: imageName,
								},
								{
									Name:  "IMAGE_TAG",
									Value: imageTag,
								},
							},
							VolumeMounts: []corev1.VolumeMount{
								{
									Name:      "data",
									MountPath: "/var/lib/containers",
								},
							},
							SecurityContext: &corev1.SecurityContext{
								Privileged: &privileged,
							},
						},
					},
					Volumes: []corev1.Volume{
						{
							Name: "data",
							VolumeSource: corev1.VolumeSource{
								HostPath: &corev1.HostPathVolumeSource{
									Path: "/root/jobdata/",
									Type: &directory,
								},
							},
						},
					},
				},
			},
		},
	}
}

func GenerateRayService(name, namespace, importPath, runtimeEnv, deploymentName string, numReplicas int32, numGpus float64, finetuneJob *finetunev1beta1.FinetuneJob, llmCheckpoint *corev1beta1.LLMCheckpoint) *rayv1.RayService {
	numReplica := &numReplicas
	numGpu := &numGpus
	enableInTreeAutoscaling := false
	workReplicas := int32(1)
	minWorkReplicas := int32(1)
	maxWorkReplicas := int32(1)
	return &rayv1.RayService{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: rayv1.RayServiceSpec{
			ServeService: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: finetuneJob.Name,
				},
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{
							Name:       "serve",
							Port:       8000,
							Protocol:   corev1.ProtocolTCP,
							TargetPort: intstr.FromInt(8000),
						},
					},
					Selector: map[string]string{
						"ray.io/node-type": "head",
					},
					Type: corev1.ServiceTypeNodePort,
				},
			},
			ServeDeploymentGraphSpec: rayv1.ServeDeploymentGraphSpec{
				ImportPath: importPath,
				RuntimeEnv: runtimeEnv,
				ServeConfigSpecs: []rayv1.ServeConfigSpec{
					{
						Name:        deploymentName,
						NumReplicas: numReplica,
						RayActorOptions: rayv1.RayActorOptionSpec{
							NumGpus: numGpu,
						},
					},
				},
			},
			RayClusterSpec: rayv1.RayClusterSpec{
				RayVersion:              "2.7.1",
				EnableInTreeAutoscaling: &enableInTreeAutoscaling,
				HeadGroupSpec: rayv1.HeadGroupSpec{
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{
									Name:            fmt.Sprintf("%s-head", finetuneJob.Name),
									Image:           *llmCheckpoint.Spec.CheckpointImage.Name,
									ImagePullPolicy: *llmCheckpoint.Spec.CheckpointImage.ImagePullPolicy,
									Env: []corev1.EnvVar{
										{
											Name:  "RAY_SERVE_ENABLE_EXPERIMENTAL_STREAMING",
											Value: "1",
										},
									},
									Ports: []corev1.ContainerPort{
										{
											Name:          "gcs-server",
											ContainerPort: 6379,
										},
										{
											Name:          "dashboard",
											ContainerPort: 8265,
										},
										{
											Name:          "client",
											ContainerPort: 10001,
										},
										{
											Name:          "serve",
											ContainerPort: 8000,
										},
									},
									Resources: corev1.ResourceRequirements{
										Limits: map[corev1.ResourceName]resource.Quantity{
											corev1.ResourceCPU:    resource.MustParse("2"),
											corev1.ResourceMemory: resource.MustParse("4Gi"),
										},
										Requests: map[corev1.ResourceName]resource.Quantity{
											corev1.ResourceCPU:    resource.MustParse("1"),
											corev1.ResourceMemory: resource.MustParse("2Gi"),
										},
									},
								},
							},
							Tolerations:  finetuneJob.Spec.ServeConfig.Tolerations,
							NodeSelector: finetuneJob.Spec.ServeConfig.NodeSelector,
						},
					},
				},
				WorkerGroupSpecs: []rayv1.WorkerGroupSpec{
					{
						Replicas:       &workReplicas,
						MinReplicas:    &minWorkReplicas,
						MaxReplicas:    &maxWorkReplicas,
						GroupName:      finetuneJob.Name,
						RayStartParams: map[string]string{},
						Template: corev1.PodTemplateSpec{
							Spec: corev1.PodSpec{
								Containers: []corev1.Container{
									{
										Name:            fmt.Sprintf("%s-work", finetuneJob.Name),
										Image:           *llmCheckpoint.Spec.CheckpointImage.Name,
										ImagePullPolicy: *llmCheckpoint.Spec.CheckpointImage.ImagePullPolicy,
										Env: []corev1.EnvVar{
											{
												Name:  "RAY_SERVE_ENABLE_EXPERIMENTAL_STREAMING",
												Value: "1",
											},
										},
										Lifecycle: &corev1.Lifecycle{
											PreStop: &corev1.LifecycleHandler{
												Exec: &corev1.ExecAction{
													Command: []string{
														"/bin/sh", "-c", "ray stop",
													},
												},
											},
										},
										Resources: corev1.ResourceRequirements{
											Limits: map[corev1.ResourceName]resource.Quantity{
												corev1.ResourceCPU:    resource.MustParse("8"),
												corev1.ResourceMemory: resource.MustParse("16Gi"),
											},
											Requests: map[corev1.ResourceName]resource.Quantity{
												corev1.ResourceCPU:    resource.MustParse("4"),
												corev1.ResourceMemory: resource.MustParse("8Gi"),
											},
										},
									},
								},
								Tolerations:  finetuneJob.Spec.ServeConfig.Tolerations,
								NodeSelector: finetuneJob.Spec.ServeConfig.NodeSelector,
							},
						},
					},
				},
			},
		},
	}

}

func GenerateBuiltInScoring(name, namespace string) *extensionv1beta1.Scoring {
	return &extensionv1beta1.Scoring{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: extensionv1beta1.ScoringSpec{
			Questions: []extensionv1beta1.Question{
				{
					Question:  "天王盖地虎",
					Reference: "小鸡炖蘑菇",
				},
			},
		},
	}
}

func GeneratePluginScoring(name, namespace, pluginName, parameters string) *extensionv1beta1.Scoring {
	return &extensionv1beta1.Scoring{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: extensionv1beta1.ScoringSpec{
			Plugin: &extensionv1beta1.Plugin{
				LoadPlugin: true,
				Name:       pluginName,
				Parameters: parameters,
			},
		},
	}
}
