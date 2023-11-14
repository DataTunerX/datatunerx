package generate

import (
	"fmt"

	finetunev1beta1 "github.com/DataTunerX/meta-server/api/finetune/v1beta1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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
func GenerateBuildImageJob(name, namespace, endpoint, accessKeyId, secretAccessKey, bucket, filePath, image, secure string) *batchv1.Job {
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
							},
							Command: []string{"bin/bash"},
							Args: []string{
								"-c",
								`buildah from docker.io/library/ubuntu
								 buildah copy containerID /local/path /path/in/container
								 buildah commit containerID your-image-name`,
							},
						},
					},
				},
			},
		},
	}
}
