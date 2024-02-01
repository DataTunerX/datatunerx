/*
Copyright 2023.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package finetune

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	corev1beta1 "github.com/DataTunerX/meta-server/api/core/v1beta1"
	extensionv1beta1 "github.com/DataTunerX/meta-server/api/extension/v1beta1"
	finetunev1beta1 "github.com/DataTunerX/meta-server/api/finetune/v1beta1"
	"github.com/DataTunerX/utility-server/logging"
	rayv1 "github.com/ray-project/kuberay/ray-operator/apis/ray/v1"
	"github.com/ray-project/kuberay/ray-operator/controllers/ray/common"
	batchv1 "k8s.io/api/batch/v1"
	"k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/remotecommand"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
)

const (
	RayVersion             = "v2.7.1"
	DefaultRequeueDuration = 3 * time.Second
	CheckpointPath         = "/home/ray/checkpoint_path"
)

var metricsExportAddress = os.Getenv("METRICS_EXPORT_ADDRESS")
var storagePath = os.Getenv("STORAGE_PATH")

// FinetuneReconciler reconciles a Finetune object
type FinetuneReconciler struct {
	client.Client
	Scheme    *runtime.Scheme
	Log       logging.Logger
	Clientset *kubernetes.Clientset
	Config    *rest.Config
}

//+kubebuilder:rbac:groups=finetune.datatunerx.io,resources=finetunes,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=finetune.datatunerx.io,resources=finetunes/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=finetune.datatunerx.io,resources=finetunes/finalizers,verbs=update

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
// TODO(user): Modify the Reconcile function to compare the state specified by
// the Finetune object against the actual cluster state, and then
// perform operations to make the cluster state reflect the state specified by
// the user.
//
// For more details, check Reconcile and its Result here:
// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.14.1/pkg/reconcile
func (r *FinetuneReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	r.Log.Infof("Reconciling Finetune: %+v", req.NamespacedName)
	finetuneInstance := &finetunev1beta1.Finetune{}

	err := r.Get(ctx, req.NamespacedName, finetuneInstance)

	if err != nil {
		if apierrors.IsNotFound(err) {
			r.Log.Infof("Finetune: %+v not found. Ignoring since object must be deleted", req.NamespacedName)
			return ctrl.Result{}, nil
		}
		// Error reading the object - requeue the request.
		r.Log.Error("Failed to get Finetune")
		return ctrl.Result{RequeueAfter: DefaultRequeueDuration}, err
	}

	if finetuneInstance.ObjectMeta.DeletionTimestamp.IsZero() {
		if !controllerutil.ContainsFinalizer(finetuneInstance, finetunev1beta1.FinetuneGroupFinalizer) {
			controllerutil.AddFinalizer(finetuneInstance, finetunev1beta1.FinetuneGroupFinalizer)
			if err := r.Update(context.Background(), finetuneInstance); err != nil {
				r.Log.Errorf("Failed to update Finetune with finalizer: %v", err)
				return ctrl.Result{RequeueAfter: DefaultRequeueDuration}, err
			}
		}
	} else {
		r.Log.Infof("Finetune: %+v is being deleted", req.NamespacedName)
		controllerutil.RemoveFinalizer(finetuneInstance, finetunev1beta1.FinetuneGroupFinalizer)
		if err := r.Update(context.Background(), finetuneInstance); err != nil {
			r.Log.Errorf("Failed to update Finetune without finalizer: %v", err)
			return ctrl.Result{RequeueAfter: DefaultRequeueDuration}, err
		}
		return ctrl.Result{RequeueAfter: DefaultRequeueDuration}, nil
	}

	if finetuneInstance.Status.State == finetunev1beta1.FinetuneSuccessful {
		r.Log.Infof("Finetune: %+v is Successful.", req.NamespacedName)
		return ctrl.Result{}, nil
	}

	if finetuneInstance.Status.State == finetunev1beta1.FinetuneFailed {
		r.Log.Infof("Finetune: %+v is Failed.", req.NamespacedName)
		return ctrl.Result{}, nil
	}

	if finetuneInstance.Status.State == "" {
		if err = r.updateFinetuneState(ctx, finetuneInstance, finetunev1beta1.FinetuneInit); err != nil {
			r.Log.Errorf("Finetune %v update state error: %s", req.NamespacedName, err)
			return ctrl.Result{RequeueAfter: DefaultRequeueDuration}, err
		}
	}

	rayJobInstance := &rayv1.RayJob{}

	err = r.Get(ctx, req.NamespacedName, rayJobInstance)
	if err != nil {
		if apierrors.IsNotFound(err) {
			r.Log.Info("RayJob not found. Create a new one.")
			err = r.createRayJob(ctx, finetuneInstance)
			if err != nil {
				r.Log.Errorf("Failed to create RayJob: %v", err)
				if err = r.updateFinetuneState(ctx, finetuneInstance, finetunev1beta1.FinetunePending); err != nil {
					r.Log.Errorf("Finetune %v update state error: %s", req.NamespacedName, err)
					return ctrl.Result{RequeueAfter: DefaultRequeueDuration}, err
				}
				return ctrl.Result{RequeueAfter: DefaultRequeueDuration}, err
			}
			return ctrl.Result{RequeueAfter: DefaultRequeueDuration}, nil
		} else {
			r.Log.Error("Failed to get RayJob")
			return ctrl.Result{RequeueAfter: DefaultRequeueDuration}, err
		}
	}

	if err = r.updateFinetuneState(ctx, finetuneInstance, finetunev1beta1.FinetuneRunning); err != nil {
		r.Log.Errorf("Finetune %v update state error: %s", req.NamespacedName, err)
		return ctrl.Result{RequeueAfter: DefaultRequeueDuration}, err
	}

	if rayJobInstance.Spec.ShutdownAfterJobFinishes == false {
		rayJobInstance.Spec.ShutdownAfterJobFinishes = true
		r.Log.Infof("RayJob %s/%s set ShutdownAfterJobFinishes true", rayJobInstance.Namespace, rayJobInstance.Name)
		if err := r.Update(ctx, rayJobInstance); err != nil {
			r.Log.Errorf("Failed to update RayJob %v: %v", req.NamespacedName, err)
			return ctrl.Result{RequeueAfter: DefaultRequeueDuration}, err
		}
	}

	// check rayjob status
	r.Log.Infof("RayJob %s/%s status is %s", rayJobInstance.Namespace, rayJobInstance.Name, rayJobInstance.Status.JobStatus)
	if rayJobInstance.Status.JobStatus == "" {
		return ctrl.Result{RequeueAfter: DefaultRequeueDuration * 10}, nil
	}

	// update rayjob info
	if finetuneInstance.Status.RayJobInfo == nil {
		rajJobInfo, err := r.getRayJobPodInfo(ctx, rayJobInstance)
		if err != nil {
			r.Log.Errorf("getRayJobPodInfo err: %s", err)
			return ctrl.Result{RequeueAfter: DefaultRequeueDuration}, err
		}
		finetuneInstance.Status.RayJobInfo = rajJobInfo

		if err = r.Status().Update(ctx, finetuneInstance); err != nil {
			r.Log.Errorf("Failed to update Finetune status %v: %v", req.NamespacedName, err)
			return ctrl.Result{RequeueAfter: DefaultRequeueDuration}, err
		}
	}

	if isJobPendingOrRunning(rayJobInstance.Status.JobStatus) {
		return ctrl.Result{RequeueAfter: DefaultRequeueDuration * 10}, nil
	}

	if isJobStoppedOrFailed(rayJobInstance.Status.JobStatus) {
		if err = r.updateFinetuneState(ctx, finetuneInstance, finetunev1beta1.FinetuneFailed); err != nil {
			r.Log.Errorf("Finetune %v update state error: %s", req.NamespacedName, err)
			return ctrl.Result{RequeueAfter: DefaultRequeueDuration}, err
		}
		return ctrl.Result{RequeueAfter: DefaultRequeueDuration}, nil
	}

	if finetuneInstance.Status.LLMCheckpoint == nil {
		headPod, err := r.getRayClusterHeadPod(ctx, rayJobInstance)
		if err != nil {
			r.Log.Errorf("getRayClusterHeadPod err: %s", err)
			return ctrl.Result{RequeueAfter: DefaultRequeueDuration}, err
		}

		checkpointPath, err := r.fetchPodFile(ctx, headPod, CheckpointPath)
		if err != nil {
			r.Log.Errorf("fetchPodFile err: %s", err)
			return ctrl.Result{RequeueAfter: DefaultRequeueDuration}, err
		}

		finetuneInstance.Status.LLMCheckpoint = &finetunev1beta1.Checkpoint{
			LLMCheckpointRef: GenerateLLMCheckpointName(finetuneInstance.Name),
			CheckpointPath:   checkpointPath,
		}
		if err = r.Status().Update(ctx, finetuneInstance); err != nil {
			r.Log.Errorf("Failed to update Finetune status %v: %v", req.NamespacedName, err)
			return ctrl.Result{RequeueAfter: DefaultRequeueDuration}, err
		}

	}

	// create llmcheckpoit
	if err := r.reconcileLLMCheckpoint(ctx, finetuneInstance); err != nil {
		r.Log.Errorf("reconcileLLMCheckpoint err: %s", err)
		return ctrl.Result{RequeueAfter: DefaultRequeueDuration}, err
	}

	if err = r.updateFinetuneState(ctx, finetuneInstance, finetunev1beta1.FinetuneSuccessful); err != nil {
		r.Log.Errorf("Finetune %v update state error: %s", req.NamespacedName, err)
		return ctrl.Result{RequeueAfter: DefaultRequeueDuration}, err
	}

	return ctrl.Result{}, nil
}

func (r *FinetuneReconciler) getRayJobPodInfo(ctx context.Context, rayJob *rayv1.RayJob) (*finetunev1beta1.RayJobInfo, error) {
	job := &batchv1.Job{}
	if err := r.Get(ctx, client.ObjectKey{Namespace: rayJob.Namespace, Name: rayJob.Name}, job); err != nil {
		return nil, err
	}

	jobPods := &v1.PodList{}
	err := r.List(ctx, jobPods, client.InNamespace(rayJob.Namespace), client.MatchingLabels{"batch.kubernetes.io/job-name": rayJob.Name})
	if err != nil {
		return nil, err
	}
	if len(jobPods.Items) == 0 {
		return nil, fmt.Errorf("RayJob: %s/%s has no pod", rayJob.Namespace, rayJob.Name)
	}
	var firstPod string
	var podStartTime *metav1.Time
	for _, pod := range jobPods.Items {
		if podStartTime == nil || podStartTime.Time.After(pod.CreationTimestamp.Time) {
			firstPod = pod.Name
			podStartTime = &pod.CreationTimestamp
		}
	}

	return &finetunev1beta1.RayJobInfo{RayJobPodName: firstPod, RayJobPodContainerName: "ray-job-submitter"}, nil
}

func (r *FinetuneReconciler) getRayClusterHeadPod(ctx context.Context, rayJob *rayv1.RayJob) (*v1.Pod, error) {
	headPods := &v1.PodList{}
	filterLabels := client.MatchingLabels{common.RayClusterLabelKey: rayJob.Status.RayClusterName, common.RayNodeTypeLabelKey: string(rayv1.HeadNode)}
	if err := r.List(ctx, headPods, client.InNamespace(rayJob.Namespace), filterLabels); err != nil {
		return nil, err
	}
	if len(headPods.Items) == 0 {
		return nil, fmt.Errorf("RayCluster: %s/%s has no head node", rayJob.Namespace, rayJob.Status.RayClusterName)
	}

	return &headPods.Items[0], nil
}

func (r *FinetuneReconciler) fetchPodFile(ctx context.Context, pod *v1.Pod, filePath string) (string, error) {
	execRequest := r.Clientset.CoreV1().RESTClient().Post().
		Resource("pods").
		Name(pod.Name).
		Namespace(pod.Namespace).
		SubResource("exec").
		VersionedParams(&v1.PodExecOptions{
			Container: pod.Spec.Containers[0].Name,
			Command:   []string{"cat", filePath},
			Stdout:    true,
			Stderr:    true,
		}, scheme.ParameterCodec)

	exec, err := remotecommand.NewSPDYExecutor(r.Config, "POST", execRequest.URL())
	if err != nil {
		return "", err
	}
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	err = exec.StreamWithContext(ctx, remotecommand.StreamOptions{
		Stdout: &stdout,
		Stderr: &stderr,
	})
	if err != nil {
		return "", err
	}
	return stdout.String(), nil
}

func (r *FinetuneReconciler) reconcileLLMCheckpoint(ctx context.Context, finetune *finetunev1beta1.Finetune) error {
	ns := finetune.Namespace
	checkpointNamespacedName := types.NamespacedName{
		Namespace: ns,
		Name:      finetune.Status.LLMCheckpoint.LLMCheckpointRef,
	}

	checkpoint := &corev1beta1.LLMCheckpoint{}
	err := r.Get(ctx, checkpointNamespacedName, checkpoint)
	if err == nil {
		return nil
	}

	if apierrors.IsNotFound(err) {
		r.Log.Infof("LLMCheckpoint: %s/%s is not found", ns, finetune.Name)

		datasetInstance, err := r.getDataset(ctx, types.NamespacedName{ns, finetune.Spec.Dataset})
		if err != nil {
			return err
		}

		hyperparameterInstance, err := r.getHyperparameter(ctx, types.NamespacedName{ns, finetune.Spec.Hyperparameter.HyperparameterRef})
		if err != nil {
			return err
		}

		llmInstance, err := r.getLLM(ctx, types.NamespacedName{ns, finetune.Spec.LLM})
		if err != nil {
			return err
		}

		llmCheckpointInstance, err := generateLLMCheckpoint(checkpointNamespacedName, finetune, datasetInstance, hyperparameterInstance, llmInstance)
		if err != nil {
			return err
		}

		// Set controller reference
		//if err := controllerutil.SetControllerReference(finetune, llmCheckpointInstance, r.Scheme); err != nil {
		//	return err
		//}

		return r.Create(ctx, llmCheckpointInstance)

	}

	return err
}

func (r *FinetuneReconciler) getLLM(ctx context.Context, namespacedName types.NamespacedName) (*corev1beta1.LLM, error) {
	llmInstance := &corev1beta1.LLM{}
	err := r.Get(ctx, namespacedName, llmInstance)
	if err != nil {
		r.Log.Errorf("Failed to get LLM %v", namespacedName)
		return nil, err
	}
	return llmInstance, nil
}

func (r *FinetuneReconciler) getDataset(ctx context.Context, namespacedName types.NamespacedName) (*extensionv1beta1.Dataset, error) {
	datasetInstance := &extensionv1beta1.Dataset{}
	err := r.Get(ctx, namespacedName, datasetInstance)
	if err != nil {
		r.Log.Errorf("Failed to get Dataset %v", namespacedName)
		return nil, err
	}
	return datasetInstance, nil
}

func (r *FinetuneReconciler) getHyperparameter(ctx context.Context, namespacedName types.NamespacedName) (*corev1beta1.Hyperparameter, error) {
	hyperparameterInstance := &corev1beta1.Hyperparameter{}
	err := r.Get(ctx, namespacedName, hyperparameterInstance)
	if err != nil {
		r.Log.Errorf("Failed to get Hyperparameter %v", namespacedName)
		return nil, err
	}
	return hyperparameterInstance, nil
}

// createRayJob will create the rayjob
func (r *FinetuneReconciler) createRayJob(ctx context.Context, finetune *finetunev1beta1.Finetune) error {
	ns := finetune.Namespace

	datasetInstance, err := r.getDataset(ctx, types.NamespacedName{ns, finetune.Spec.Dataset})
	if err != nil {
		finetune.Status.State = finetunev1beta1.FinetunePending
		if err := r.Status().Update(ctx, finetune); err != nil {
			return err
		}
		return err
	}

	hyperparameterInstance, err := r.getHyperparameter(ctx, types.NamespacedName{ns, finetune.Spec.Hyperparameter.HyperparameterRef})
	if err != nil {
		finetune.Status.State = finetunev1beta1.FinetunePending
		if err := r.Status().Update(ctx, finetune); err != nil {
			return err
		}
		return err
	}

	newParameters := updateHyperparameters(&hyperparameterInstance.Spec.Parameters, finetune.Spec.Hyperparameter.Overrides)
	r.Log.Debugf("newParameters: %+v", newParameters)
	rayJobEntrypoint, err := getRayJobEntrypoint(ctx, finetune, datasetInstance, newParameters)
	if err != nil {
		return err
	}

	r.Log.Info("create ray cluster")
	rayJobInstance, err := generateRayJob(ctx, &types.NamespacedName{ns, finetune.Name}, rayJobEntrypoint, finetune)
	if err != nil {
		return err
	}

	// Set controller reference
	if err := controllerutil.SetControllerReference(finetune, rayJobInstance, r.Scheme); err != nil {
		return err
	}

	return r.Create(ctx, rayJobInstance)
}

// updateFinetuneState is a method of the FinetuneReconciler struct.
// It updates the state of a Finetune instance and logs the new state.
//
// Parameters:
// ctx: The context within which the function is called. Used for timeout and cancellation signals.
// instance: The Finetune instance whose state is to be updated.
// finetuneState: The new state to be set for the Finetune instance.
//
// Returns:
// error: An error object that describes an error that occurred during the function's execution. Returns nil if the function executed successfully.
func (r *FinetuneReconciler) updateFinetuneState(ctx context.Context, instance *finetunev1beta1.Finetune, finetuneState finetunev1beta1.FinetuneState) error {
	// If the current state is the same as the new state, return nil
	if instance.Status.State == finetuneState {
		return nil
	}
	// Update the state of the Finetune instance
	instance.Status.State = finetuneState
	// Log the new state
	r.Log.Infof("Update Finetune CR Status.State: %s", finetuneState)
	// Update the status of the Finetune instance in the Kubernetes API and return any error that occurs
	return r.Status().Update(ctx, instance)
}

func getRayJobEntrypoint(ctx context.Context, finetune *finetunev1beta1.Finetune, dataset *extensionv1beta1.Dataset, parameters *corev1beta1.Parameters) (string, error) {
	// TODO check parameters include blank
	replicas := int32(finetune.Spec.Node)
	if replicas <= 0 {
		replicas = 1
	}
	entrypoint := []string{"python"}
	entrypoint = append(entrypoint, "/tuning/train.py")

	finetunePath := finetune.Spec.Image.Path
	if finetunePath == "" {
		return "", fmt.Errorf("%s/%s: finetune.Spec.Image.Path is required", finetune.Namespace, finetune.Name)
	}
	entrypoint = append(entrypoint, "--model_name_or_path", finetunePath)

	entrypoint = append(entrypoint, "--train_path", dataset.Spec.DatasetMetadata.DatasetInfo.Subsets[0].Splits.Train.File)

	if dataset.Spec.DatasetMetadata.DatasetInfo.Subsets[0].Splits.Validate != nil && dataset.Spec.DatasetMetadata.DatasetInfo.Subsets[0].Splits.Validate.File != "" {
		entrypoint = append(entrypoint, "--evaluation_path", dataset.Spec.DatasetMetadata.DatasetInfo.Subsets[0].Splits.Validate.File)
	}

	featuresMapJson, err := getFeaturesMapJson(dataset.Spec.DatasetMetadata.DatasetInfo.Features)
	if err != nil {
		return "", err
	}
	if featuresMapJson != "" {
		entrypoint = append(entrypoint, "--columns", strconv.Quote(featuresMapJson))
	}

	entrypoint = append(entrypoint, "--output_dir", "result")
	entrypoint = append(entrypoint, "--deepspeed", "/tuning/ds_config.json")
	entrypoint = append(entrypoint, "--lora_target", "q_proj,v_proj")
	entrypoint = append(entrypoint, "--lr_scheduler_type", string(parameters.Scheduler))
	entrypoint = append(entrypoint, "--optim", string(parameters.Optimizer))

	quantization := ""
	if parameters.Int8 {
		quantization = "int8"
	} else if parameters.Int4 {
		quantization = "int4"
	}
	if quantization != "" {
		entrypoint = append(entrypoint, "--quantization", quantization)
	}

	entrypoint = append(entrypoint, "--lora_r", parameters.LoRA_R)
	entrypoint = append(entrypoint, "--lora_alpha", parameters.LoRA_Alpha)
	entrypoint = append(entrypoint, "--lora_dropout", parameters.LoRA_Dropout)
	entrypoint = append(entrypoint, "--learning_rate", parameters.LearningRate)
	entrypoint = append(entrypoint, "--num_train_epochs", fmt.Sprintf("%d", parameters.Epochs))
	entrypoint = append(entrypoint, "--block_size", fmt.Sprintf("%d", parameters.BlockSize))
	entrypoint = append(entrypoint, "--per_device_train_batch_size ", fmt.Sprintf("%d", parameters.BatchSize))
	entrypoint = append(entrypoint, "--warmup_ratio", parameters.WarmupRatio)
	entrypoint = append(entrypoint, "--weight_decay", parameters.WeightDecay)
	entrypoint = append(entrypoint, "--gradient_accumulation_steps", fmt.Sprintf("%d", parameters.GradAccSteps))
	entrypoint = append(entrypoint, "--fp16", fmt.Sprintf("%t", parameters.FP16))
	entrypoint = append(entrypoint, "--num_workers", fmt.Sprintf("%d", replicas))
	entrypoint = append(entrypoint, "--storage_path", storagePath)

	if metricsExportAddress != "" {
		entrypoint = append(entrypoint, "--metrics_export_address", metricsExportAddress)
		entrypoint = append(entrypoint, "--uid", fmt.Sprintf("%s", finetune.UID))

	}
	return strings.Join(entrypoint, " "), nil
}

func generateRayJob(ctx context.Context, namespacedName *types.NamespacedName, entrypoint string, finetune *finetunev1beta1.Finetune) (*rayv1.RayJob, error) {
	replicas := int32(finetune.Spec.Node)
	if replicas <= 0 {
		replicas = 1
	}
	if finetune.Spec.Image.Name == "" {
		return nil, fmt.Errorf("%s/%s: finetune.Spec.Image.Name is required", finetune.Namespace, finetune.Name)
	}

	var rayJobInstance = &rayv1.RayJob{
		TypeMeta: metav1.TypeMeta{
			Kind:       "RayJob",
			APIVersion: "ray.io/v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      namespacedName.Name,
			Namespace: namespacedName.Namespace,
		},
		Spec: rayv1.RayJobSpec{
			//ShutdownAfterJobFinishes: true,
			Entrypoint: entrypoint,
			RayClusterSpec: &rayv1.RayClusterSpec{
				RayVersion: RayVersion,
				HeadGroupSpec: rayv1.HeadGroupSpec{
					ServiceType:   "NodePort",
					HeadService:   nil,
					EnableIngress: nil,
					Replicas:      nil,
					RayStartParams: map[string]string{
						"dashboard-host": "0.0.0.0",
						"num-gpus":       "0",
					},
					Template: v1.PodTemplateSpec{
						Spec: v1.PodSpec{
							Containers: []v1.Container{
								v1.Container{
									Name:            "ray-head",
									Image:           finetune.Spec.Image.Name,
									ImagePullPolicy: finetune.Spec.Image.ImagePullPolicy,
									Ports: []v1.ContainerPort{
										v1.ContainerPort{
											Name:          "gcs-server",
											ContainerPort: 6379,
										},
										v1.ContainerPort{
											Name:          "dashboard",
											ContainerPort: 8265,
										},
										v1.ContainerPort{
											Name:          "client",
											ContainerPort: 10001,
										},
									},
								},
							},
						},
					},
				},
				WorkerGroupSpecs: []rayv1.WorkerGroupSpec{rayv1.WorkerGroupSpec{
					GroupName:      "finetune-group",
					Replicas:       &replicas,
					RayStartParams: map[string]string{},
					Template: v1.PodTemplateSpec{
						Spec: v1.PodSpec{
							Containers: []v1.Container{
								v1.Container{
									Name:            "ray-worker",
									Image:           finetune.Spec.Image.Name,
									ImagePullPolicy: finetune.Spec.Image.ImagePullPolicy,
									Lifecycle: &v1.Lifecycle{
										PreStop: &v1.LifecycleHandler{
											Exec: &v1.ExecAction{
												Command: []string{
													"/bin/sh", "-c", "ray stop",
												},
											},
										},
									},
									Resources: v1.ResourceRequirements{
										Requests: v1.ResourceList{
											"nvidia.com/gpu": resource.MustParse("1"),
										},
										Limits: v1.ResourceList{
											"nvidia.com/gpu": resource.MustParse("1"),
										},
									},
								},
							},
						},
					},
					ScaleStrategy: rayv1.ScaleStrategy{},
				}},
				EnableInTreeAutoscaling: nil,
				AutoscalerOptions:       nil,
				HeadServiceAnnotations:  nil,
			},
			ClusterSelector: nil,
			Suspend:         false,
		},
	}
	return rayJobInstance, nil
}

func generateLLMCheckpoint(checkpointNamespacedName types.NamespacedName, finetune *finetunev1beta1.Finetune, dataset *extensionv1beta1.Dataset, hyperparameter *corev1beta1.Hyperparameter, llm *corev1beta1.LLM) (*corev1beta1.LLMCheckpoint, error) {
	if finetune.Status.LLMCheckpoint == nil || finetune.Status.LLMCheckpoint.CheckpointPath == "" {
		return nil, fmt.Errorf("CheckpointPath is nil")
	}

	var llmCheckpointInstance = &corev1beta1.LLMCheckpoint{
		TypeMeta: metav1.TypeMeta{
			Kind:       "LLMCheckpoint",
			APIVersion: "core.datatunerx.io/v1beta1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      checkpointNamespacedName.Name,
			Namespace: checkpointNamespacedName.Namespace,
		},
		Spec: corev1beta1.LLMCheckpointSpec{
			LLM: &corev1beta1.LLMRefSpec{
				LLMRef: finetune.Spec.LLM,
				Spec:   llm.Spec.DeepCopy(),
			},
			Dataset: &corev1beta1.DatasetRefSpec{
				DatasetRef: finetune.Spec.Dataset,
				Spec:       dataset.Spec.DeepCopy(),
			},
			Hyperparameter: &corev1beta1.HyperparameterRefSpec{
				HyperparameterRef: finetune.Spec.Hyperparameter.HyperparameterRef,
				Spec:              hyperparameter.Spec.DeepCopy(),
			},
			Image:      &finetune.Spec.Image,
			Checkpoint: finetune.Status.LLMCheckpoint.CheckpointPath,
		},
	}
	return llmCheckpointInstance, nil
}

func getFeaturesMapJson(features []extensionv1beta1.Feature) (string, error) {
	if features == nil {
		return "", nil
	}
	featuresMap := make(map[string]string)
	for _, feature := range features {
		if feature.Name == "instruction" {
			featuresMap["instruction"] = feature.MapTo
			continue
		}
		if feature.Name == "response" {
			featuresMap["response"] = feature.MapTo
		}
	}

	if len(featuresMap) == 0 {
		return "", nil
	}

	jsonData, err := json.Marshal(featuresMap)
	if err != nil {
		return "", err
	}

	return string(jsonData), nil
}

func updateHyperparameters(parameters *corev1beta1.Parameters, overrides *finetunev1beta1.Parameters) *corev1beta1.Parameters {
	newParameters := parameters.DeepCopy()

	if overrides == nil {
		return newParameters
	}

	if overrides.Scheduler != "" {
		newParameters.Scheduler = overrides.Scheduler
	}

	if overrides.Optimizer != "" {
		newParameters.Optimizer = overrides.Optimizer
	}

	if overrides.Int4 != nil {
		newParameters.Int4 = *overrides.Int4
	}

	if overrides.Int8 != nil {
		newParameters.Int8 = *overrides.Int8
	}

	if overrides.LoRA_R != nil {
		newParameters.LoRA_R = *overrides.LoRA_R
	}

	if overrides.LoRA_Alpha != nil {
		newParameters.LoRA_Alpha = *overrides.LoRA_Alpha
	}

	if overrides.LoRA_Dropout != nil {
		newParameters.LoRA_Dropout = *overrides.LoRA_Dropout
	}

	if overrides.LearningRate != nil {
		newParameters.LearningRate = *overrides.LearningRate
	}

	if overrides.Epochs != 0 {
		newParameters.Epochs = overrides.Epochs
	}

	if overrides.BlockSize != 0 {
		newParameters.BlockSize = overrides.BlockSize
	}

	if overrides.BatchSize != 0 {
		newParameters.BatchSize = overrides.BatchSize
	}

	if overrides.WarmupRatio != nil {
		newParameters.WarmupRatio = *overrides.WarmupRatio
	}

	if overrides.WeightDecay != nil {
		newParameters.WeightDecay = *overrides.WeightDecay
	}

	if overrides.GradAccSteps != 0 {
		newParameters.GradAccSteps = overrides.GradAccSteps
	}

	if overrides.TrainerType != nil {
		newParameters.TrainerType = *overrides.TrainerType
	}

	if overrides.PEFT != nil {
		newParameters.PEFT = *overrides.PEFT
	}

	if overrides.FP16 != nil {
		newParameters.FP16 = *overrides.FP16
	}

	return newParameters
}

// isJobPendingOrRunning indicates whether the job is running.
func isJobPendingOrRunning(status rayv1.JobStatus) bool {
	return (status == rayv1.JobStatusPending) || (status == rayv1.JobStatusRunning)
}

// isJobPendingOrRunning indicates whether the job is running.
func isJobStoppedOrFailed(status rayv1.JobStatus) bool {
	return (status == rayv1.JobStatusStopped) || (status == rayv1.JobStatusFailed)
}

// GenerateLLMCheckpointName generates a LLMCheckpoint name from Finetune name
func GenerateLLMCheckpointName(finetuneName string) string {
	return fmt.Sprintf("%s-%s", finetuneName, rand.String(5))
}

// SetupWithManager sets up the controller with the Manager.
func (r *FinetuneReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&finetunev1beta1.Finetune{}).
		Owns(&rayv1.RayJob{}).
		Owns(&corev1beta1.LLMCheckpoint{}).
		WithOptions(controller.Options{CacheSyncTimeout: 10 * time.Second}).
		Complete(r)
}
