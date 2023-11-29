package generate

import (
	"fmt"

	"github.com/DataTunerX/finetune-experiment-controller/pkg/config"
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

const (
	defaultFinetuneImage = "rayproject/ray271-llama2-7b-finetune:20231124"
	// todo llm file path
	defaultFinetuneCodePath = "/tmp/llama2-7b/"
	zeroString              = ""

	defaultBuildImageJobContainerName = "imagebuild"
	defaultBuildImageJobImage         = "release.daocloud.io/datatunerx/buildimage:v0.0.1"
)

func GenerateFinetune(finetuneJob *finetunev1beta1.FinetuneJob) *finetunev1beta1.Finetune {
	if finetuneJob.Spec.FineTune.Name == "" {
		finetuneJob.Spec.FineTune.Name = fmt.Sprintf("%s-%s", finetuneJob.Name, "finetune")
	}
	if finetuneJob.Spec.FineTune.FinetuneSpec.Node <= 0 {
		finetuneJob.Spec.FineTune.FinetuneSpec.Node = 2
	}
	finetune := &finetunev1beta1.Finetune{
		ObjectMeta: metav1.ObjectMeta{
			Name:      finetuneJob.Spec.FineTune.Name,
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
	if finetuneJob.Spec.FineTune.FinetuneSpec.Image.Name == zeroString {
		finetune.Spec.Image.Name = defaultFinetuneImage
	}
	if finetuneJob.Spec.FineTune.FinetuneSpec.Image.Path == zeroString {
		finetune.Spec.Image.Path = defaultFinetuneCodePath
	}
	return finetune
}

// todo(tigerK) add build image job
func GenerateBuildImageJob(name, namespace, filePath string) *batchv1.Job {
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
							Name:  defaultBuildImageJobContainerName,
							Image: defaultBuildImageJobImage,
							Env: []corev1.EnvVar{
								{
									Name:  "S3_ENDPOINT",
									Value: config.GetS3Endpoint(),
								},
								{Name: "S3_ACCESSKEYID",
									Value: config.GetS3AccesskeyId(),
								},
								{
									Name:  "S3_SECRETACCESSKEY",
									Value: config.GetS3ESecretAccessKey(),
								},
								{
									Name:  "S3_BUCKET",
									Value: config.GetS3Bucket(),
								},
								{
									Name:  "S3_FILEPATH",
									Value: filePath,
								},
								{
									Name:  "S3_SECURE",
									Value: config.GetSecure(),
								},
								{
									Name:  "MOUNT_PATH",
									Value: config.GetMountPath(),
								},
								{
									Name:  "REGISTRY_URL",
									Value: config.GetRegistryUrl(),
								},
								{
									Name:  "REPOSITORY_NAME",
									Value: config.GetRepositoryName(),
								},
								{
									Name:  "USERNAME",
									Value: config.GetUserName(),
								},
								{
									Name:  "PASSWORD",
									Value: config.GetPassword(),
								},
								{
									Name:  "IMAGE_NAME",
									Value: config.GetImageName(),
								},
								{
									Name:  "IMAGE_TAG",
									Value: config.GetImageTag(),
								},
							},
							VolumeMounts: []corev1.VolumeMount{
								{
									Name:      "data",
									MountPath: config.GetMountPath(),
								},
							},
							SecurityContext: &corev1.SecurityContext{
								Privileged: &privileged,
							},
						},
					},
					RestartPolicy: corev1.RestartPolicyNever,
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
	// todo(tigerK) hardcode for rubbish
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
					RayStartParams: map[string]string{
						"dashboard-host": "0.0.0.0",
						"num-gpus":       "0",
					},
					ServiceType: corev1.ServiceTypeNodePort,
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{
									Name:            fmt.Sprintf("%s-head", finetuneJob.Name),
									Image:           *llmCheckpoint.Spec.CheckpointImage.Name,
									ImagePullPolicy: corev1.PullAlways,
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
											corev1.ResourceMemory: resource.MustParse("16Gi"),
										},
										Requests: map[corev1.ResourceName]resource.Quantity{
											corev1.ResourceCPU:    resource.MustParse("1"),
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
										ImagePullPolicy: corev1.PullAlways,
										Env: []corev1.EnvVar{
											{
												Name:  "RAY_SERVE_ENABLE_EXPERIMENTAL_STREAMING",
												Value: "1",
											},
											{
												Name:  "BASE_MODEL_DIR",
												Value: llmCheckpoint.Spec.CheckpointImage.LLMPath,
											},
											{
												Name:  "CHECKPOINT_DIR",
												Value: llmCheckpoint.Spec.CheckpointImage.CheckPointPath,
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
												corev1.ResourceMemory: resource.MustParse("64Gi"),
											},
											Requests: map[corev1.ResourceName]resource.Quantity{
												corev1.ResourceCPU:    resource.MustParse("4"),
												corev1.ResourceMemory: resource.MustParse("32Gi"),
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

func GenerateBuiltInScoring(name, namespace, inference string) *extensionv1beta1.Scoring {
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
			InferenceService: inference,
		},
	}
}

func GeneratePluginScoring(name, namespace, pluginName, parameters, inference string) *extensionv1beta1.Scoring {
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
			InferenceService: inference,
		},
	}
}
