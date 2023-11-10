package generate

import (
	"fmt"

	finetunev1beta1 "github.com/DataTunerX/meta-server/api/finetune/v1beta1"
	batchv1 "k8s.io/api/batch/v1"
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
func GenerateBuildImageJob() *batchv1.Job {
	return &batchv1.Job{}

}
