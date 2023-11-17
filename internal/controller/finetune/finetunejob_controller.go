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
	"context"
	"fmt"
	"reflect"
	"time"

	"github.com/DataTunerX/finetune-experiment-controller/pkg/config"
	"github.com/DataTunerX/finetune-experiment-controller/pkg/util/generate"
	"github.com/DataTunerX/finetune-experiment-controller/pkg/util/handlererr"
	corev1beta1 "github.com/DataTunerX/meta-server/api/core/v1beta1"
	extensionv1beta1 "github.com/DataTunerX/meta-server/api/extension/v1beta1"
	finetunev1beta1 "github.com/DataTunerX/meta-server/api/finetune/v1beta1"
	"github.com/DataTunerX/utility-server/logging"
	rayv1 "github.com/ray-project/kuberay/ray-operator/apis/ray/v1"
	batchv1 "k8s.io/api/batch/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
	"sigs.k8s.io/controller-runtime/pkg/source"
)

// FinetuneJobReconciler reconciles a FinetuneJob object
type FinetuneJobReconciler struct {
	client.Client
	Scheme *runtime.Scheme
	Log    logging.Logger
}

//+kubebuilder:rbac:groups=finetune.datatunerx.io,resources=finetunejobs,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=finetune.datatunerx.io,resources=finetunejobs/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=finetune.datatunerx.io,resources=finetunejobs/finalizers,verbs=update

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
// TODO(user): Modify the Reconcile function to compare the state specified by
// the FinetuneJob object against the actual cluster state, and then
// perform operations to make the cluster state reflect the state specified by
// the user.
//
// For more details, check Reconcile and its Result here:
// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.14.1/pkg/reconcile
func (r *FinetuneJobReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	// todo(tigerK) This reconcile contains a lot of tested strings that need to be optimised after running through the process
	r.Log.Infof("Start reconcile finetuneJob: %s/%s,", req.Name, req.Namespace)
	finetuneJob := &finetunev1beta1.FinetuneJob{}
	if err := r.Get(ctx, req.NamespacedName, finetuneJob); err != nil {
		if errors.IsNotFound(err) {
			r.Log.Warnf("FinetuneJob resource not found. Ignoring since object must be deleted.")
			return handlererr.HandlerErr(nil)
		}
		r.Log.Errorf("Failed get finetuneJob: %s/%s, Err: %v", req.Name, req.Namespace, err)
		return handlererr.HandlerErr(err)
	}

	if finetuneJob.GetDeletionTimestamp() != nil {
		r.Log.Infof("Delete finetuneJob: %s/%s", req.Name, req.Namespace)
		if controllerutil.ContainsFinalizer(finetuneJob, finetunev1beta1.FinetuneGroupFinalizer) {
			// todo cleaner
			controllerutil.RemoveFinalizer(finetuneJob, finetunev1beta1.FinetuneGroupFinalizer)
			if err := r.Update(ctx, finetuneJob); err != nil {
				r.Log.Errorf("Remove finalizer failed: %s/%s, Err: %v", req.Name, req.Namespace, err)
				return handlererr.HandlerErr(err)
			}
		}
		return handlererr.HandlerErr(nil)
	}
	if !controllerutil.ContainsFinalizer(finetuneJob, finetunev1beta1.FinetuneGroupFinalizer) {
		controllerutil.AddFinalizer(finetuneJob, finetunev1beta1.FinetuneGroupFinalizer)
		err := r.Update(ctx, finetuneJob)
		if err != nil {
			r.Log.Errorf("Add finalizer failed: %s/%s, %v", req.Name, req.Namespace, err)
			return handlererr.HandlerErr(err)
		}
	}

	// Phase I of the fine-tuning exercise.
	// Checking the existence of external dependent resources.
	llm := &corev1beta1.LLM{}
	if err := r.Client.Get(ctx, types.NamespacedName{Name: finetuneJob.Spec.FineTune.FinetuneSpec.LLM, Namespace: req.Namespace}, llm); err != nil {
		r.Log.Errorf("Failed get llm: %s/%s, err: %v", finetuneJob.Spec.FineTune.FinetuneSpec.LLM, req.Namespace, err)
		return handlererr.HandlerErr(err)
	}
	hyperparameter := &corev1beta1.Hyperparameter{}
	if err := r.Client.Get(ctx, types.NamespacedName{Name: finetuneJob.Spec.FineTune.FinetuneSpec.Hyperparameter.HyperparameterRef, Namespace: req.Namespace}, hyperparameter); err != nil {
		r.Log.Errorf("Failed get hyperparameter: %s/%s, err: %v", finetuneJob.Spec.FineTune.FinetuneSpec.Hyperparameter.HyperparameterRef, req.Namespace, err)
		return handlererr.HandlerErr(err)
	}
	dataset := &extensionv1beta1.Dataset{}
	if err := r.Client.Get(ctx, types.NamespacedName{Name: finetuneJob.Spec.FineTune.FinetuneSpec.Dataset, Namespace: req.Namespace}, dataset); err != nil {
		r.Log.Errorf("Failed get dataset: %s/%s, err: %v", finetuneJob.Spec.FineTune.FinetuneSpec.Dataset, req.Namespace, err)
		return handlererr.HandlerErr(err)
	}
	// Phase II of the fine-tuning exercise.
	// Generate finetune CR.
	finetune := generate.GenerateFinetune(finetuneJob)
	existFinetune := &finetunev1beta1.Finetune{}
	if err := r.Get(ctx, types.NamespacedName{
		Name:      finetune.Name,
		Namespace: finetune.Namespace,
	}, existFinetune); err != nil {
		if errors.IsNotFound(err) {
			r.Log.Infof("Starting to send down finetune: %s/%s.", finetune.Name, finetune.Namespace)
			if err := ctrl.SetControllerReference(finetuneJob, finetune, r.Scheme); err != nil {
				r.Log.Errorf("Set owner failed: %v", err)
				return handlererr.HandlerErr(err)
			}
			if err := r.Client.Create(ctx, finetune); err != nil {
				if !errors.IsAlreadyExists(err) {
					r.Log.Errorf("Create finetune failed: %v", err)
					return handlererr.HandlerErr(err)
				}
			}
			r.Log.Infof("Update finetuneJob %s/%s status %s.", req.Name, req.Namespace, finetunev1beta1.FinetuneJobFinetune)
			finetuneJob.Status.State = finetunev1beta1.FinetuneJobFinetune
			finetuneJob.Status.FinetuneState = existFinetune.Status.State
			if err := r.Client.Status().Update(ctx, finetuneJob); err != nil {
				r.Log.Errorf("Update finetuneJob status failed: %v", err)
				return handlererr.HandlerErr(err)
			}
			return ctrl.Result{RequeueAfter: 3 * time.Second}, nil
		}
	}
	// Phase III of the fine-tuning exercise.
	// Update finetunejob status.
	if existFinetune.Status.State == finetunev1beta1.FinetuneSuccessful && finetuneJob.Status.State != finetunev1beta1.FinetuneJobBuildImage {
		// Get llmCheckpoint Cr
		llmCheckpoint := &corev1beta1.LLMCheckpoint{}
		if err := r.Get(ctx, types.NamespacedName{Name: existFinetune.Status.LLMCheckpoint, Namespace: req.Namespace}, llmCheckpoint); err != nil {
			r.Log.Errorf("Get llmCheckpoint %s/%s failed, err: %v", existFinetune.Status.LLMCheckpoint, req.Namespace, err)
			return handlererr.HandlerErr(err)
		}
		// build llmCheckpoint image server. job
		endpoint := config.GetS3Endpoint()
		accesskeyId := config.GetS3AccesskeyId()
		accessSecretkey := config.GetS3ESecretAccessKey()
		bucket := config.GetS3Bucket()
		filePath := llmCheckpoint.Spec.Checkpoint
		secure := config.GetSecure()
		image := "release.daocloud.io/datatunerx/buildimage:v0.0.1"
		userName := config.GetUserName()
		password := config.GetPassword()
		repositoryName := config.GetRepositoryName()
		registryUrl := config.GetRegistryUrl()
		mountPath := config.GetMountPath()
		imageTag := config.GetImageTag()
		imageName := config.GetImageName()
		buildImageName := fmt.Sprintf("%s-buildimage", finetuneJob.Name)
		buildImageJob := generate.GenerateBuildImageJob(buildImageName, finetuneJob.Namespace,
			endpoint, accesskeyId, accessSecretkey, bucket, filePath, image, secure, mountPath, registryUrl, repositoryName, userName, password, imageName, imageTag)
		if err := r.Client.Create(ctx, buildImageJob); err != nil {
			if !errors.IsAlreadyExists(err) {
				r.Log.Errorf("Create job %s/%s failed, err: %v", buildImageJob.Name, buildImageJob.Namespace, err)
				return handlererr.HandlerErr(err)
			}
		}
		finetuneJob.Status.State = finetunev1beta1.FinetuneJobBuildImage
		finetuneJob.Status.FinetuneState = existFinetune.Status.State
		if err := r.Client.Status().Update(ctx, finetuneJob); err != nil {
			r.Log.Errorf("Update finetune status failed: %v", err)
			return handlererr.HandlerErr(err)
		}
	}

	if finetuneJob.Status.State == finetunev1beta1.FinetuneJobBuildImage {
		jobName := fmt.Sprintf("%s-buildimage", finetuneJob.Name)
		buildImageJob := &batchv1.Job{}
		if err := r.Get(ctx, types.NamespacedName{Namespace: finetuneJob.Namespace, Name: jobName}, buildImageJob); err != nil {
			if errors.IsNotFound(err) {
				r.Log.Errorf("Job %s/%s not found, err: %v", jobName, finetuneJob.Namespace, err)
				return handlererr.HandlerErr(err)
			}
			return handlererr.HandlerErr(err)
		}
		llmCheckpoint := &corev1beta1.LLMCheckpoint{}
		if err := r.Get(ctx, types.NamespacedName{Name: existFinetune.Status.LLMCheckpoint, Namespace: req.Namespace}, llmCheckpoint); err != nil {
			r.Log.Errorf("Get llmCheckpoint %s/%s failed, err: %v", existFinetune.Status.LLMCheckpoint, req.Namespace, err)
			return handlererr.HandlerErr(err)
		}
		if buildImageJob.Status.CompletionTime != nil {
			r.Log.Infof("Build image success, start update llmCheckpoint %s/%s", llmCheckpoint.Name, llmCheckpoint.Namespace)
			// todo(tigerK) update llmCheckpoint spec.checkpointimage
			r.Log.Infof("Update llmCheckpoint status successful, start send serve")
			rayServiceName := fmt.Sprintf("%s", finetuneJob.Name)
			importPath := fmt.Sprintf("%s.deployment", "test")
			runtimeEnv := ""
			deploymentName := "testDeploymentName"
			rayService := generate.GenerateRayService(rayServiceName,
				finetuneJob.Namespace, importPath, runtimeEnv, deploymentName,
				int32(1), float64(1), finetuneJob, llmCheckpoint)
			if err := ctrl.SetControllerReference(finetuneJob, rayService, r.Scheme); err != nil {
				r.Log.Errorf("Set owner failed: %v", err)
				return handlererr.HandlerErr(err)
			}
			if err := r.Create(ctx, rayService); err != nil {
				if !errors.IsAlreadyExists(err) {
					r.Log.Errorf("Create rayService %s/%s failed: %v", rayServiceName, finetuneJob.Namespace, err)
					return handlererr.HandlerErr(err)
				}
			}
			r.Log.Infof("Send serve successful")
		}
		finetuneJob.Status.State = finetunev1beta1.FinetuneJobServe
		finetuneJob.Status.FinetuneState = existFinetune.Status.State
		finetuneJob.Status.Result = &finetunev1beta1.FinetuneJobResult{
			ModelExportResult: true,
			Image:             *llmCheckpoint.Spec.CheckpointImage.Name,
		}
		if err := r.Client.Status().Update(ctx, finetuneJob); err != nil {
			r.Log.Errorf("Update finetuneJob status failed: %v", err)
			return handlererr.HandlerErr(err)
		}
	}

	if finetuneJob.Status.State == finetunev1beta1.FinetuneJobServe {
		rayServiceName := fmt.Sprintf("%s", finetuneJob.Name)
		rayService := &rayv1.RayService{}
		if err := r.Get(ctx, types.NamespacedName{
			Name:      rayServiceName,
			Namespace: finetuneJob.Namespace,
		}, rayService); err != nil {
			r.Log.Errorf("Get finetuneJob failed: %v", err)
			return handlererr.HandlerErr(err)
		}
		if rayService.Status.ServiceStatus == rayv1.Running {
			//serveNodePort := rayService.Status.ActiveServiceStatus.RayClusterStatus.Endpoints["serve"]
			//dashboardNodePort := rayService.Status.ActiveServiceStatus.RayClusterStatus.Endpoints["dashboard"]
			finetuneJob.Status.Result.Serve = fmt.Sprintf("%s.%s.svc:%s", finetuneJob.Name, finetuneJob.Namespace, "8000")
			finetuneJob.Status.Result.Dashboard = fmt.Sprintf("%s.%s.svc:%s", finetuneJob.Name, finetuneJob.Namespace, "8265")
			if err := r.Client.Status().Update(ctx, finetuneJob); err != nil {
				r.Log.Errorf("Update finetuneJob status failed: %v", err)
				return handlererr.HandlerErr(err)
			}
			scoringName := fmt.Sprintf("%s-scoring", finetuneJob.Name)
			if finetuneJob.Spec.ScoringConfig == nil {
				scoring := generate.GenerateBuiltInScoring(scoringName, finetuneJob.Namespace)
				if err := ctrl.SetControllerReference(finetuneJob, scoring, r.Scheme); err != nil {
					r.Log.Errorf("Set owner failed: %v", err)
					return handlererr.HandlerErr(err)
				}
				if err := r.Create(ctx, scoring); err != nil {
					if !errors.IsAlreadyExists(err) {
						r.Log.Errorf("Create scoring %s/%s failed: %v", scoringName, finetuneJob.Namespace, err)
						return handlererr.HandlerErr(err)
					}
				}
				return handlererr.HandlerErr(nil)
			}
			scoring := generate.GeneratePluginScoring(scoringName, finetuneJob.Namespace, finetuneJob.Spec.ScoringConfig.Name, finetuneJob.Spec.ScoringConfig.Parameters)
			if err := ctrl.SetControllerReference(finetuneJob, scoring, r.Scheme); err != nil {
				r.Log.Errorf("Set owner failed: %v", err)
				return handlererr.HandlerErr(err)
			}
			if err := r.Create(ctx, scoring); err != nil {
				if !errors.IsAlreadyExists(err) {
					r.Log.Errorf("Create scoring %s/%s failed: %v", scoringName, finetuneJob.Namespace, err)
					return handlererr.HandlerErr(err)
				}
			}
		}
	}
	scoringName := fmt.Sprintf("%s-scoring", finetuneJob.Name)
	scoring := &extensionv1beta1.Scoring{}
	if err := r.Get(ctx, types.NamespacedName{
		Name:      scoringName,
		Namespace: finetuneJob.Namespace,
	}, scoring); err != nil {
		if !errors.IsNotFound(err) {
			r.Log.Errorf("Get scoring %s/%s failed: %v", scoringName, finetuneJob.Namespace, err)
			return handlererr.HandlerErr(err)
		}
	}

	// todo(tigerK) get scoring result, update finetuneJob status
	if scoring != nil {

	}

	// Phase IIII of the fine-tuning exercise.
	// Check finetune cr status, if finetune cr status is SUCCESSFUL, start next
	return handlererr.HandlerErr(nil)
}

// SetupWithManager sets up the controller with the Manager.
func (r *FinetuneJobReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&finetunev1beta1.FinetuneJob{}, builder.WithPredicates(predicate.Funcs{
			UpdateFunc: func(updateEvent event.UpdateEvent) bool {
				oldFinetuneJob := updateEvent.ObjectOld.(*finetunev1beta1.FinetuneJob)
				newFinetuneJob := updateEvent.ObjectNew.(*finetunev1beta1.FinetuneJob)
				if !reflect.DeepEqual(oldFinetuneJob.Spec, newFinetuneJob.Spec) ||
					!newFinetuneJob.GetDeletionTimestamp().IsZero() {
					return true
				}
				return false
			},
			DeleteFunc: func(deleteEvent event.DeleteEvent) bool {
				return false
			},
		})).
		Watches(&source.Kind{Type: &finetunev1beta1.Finetune{}}, &handler.EnqueueRequestForOwner{
			OwnerType:    &finetunev1beta1.FinetuneJob{},
			IsController: true,
		}, builder.WithPredicates(predicate.Funcs{
			UpdateFunc: func(updateEvent event.UpdateEvent) bool {
				oldFinetune := updateEvent.ObjectOld.(*finetunev1beta1.Finetune)
				newFinetune := updateEvent.ObjectNew.(*finetunev1beta1.Finetune)
				if oldFinetune.Status.State != newFinetune.Status.State {
					r.Log.Infof("Get finetun %s/%s update event oldStatus: %s, newStatus: %s, skip", oldFinetune.Name, oldFinetune.Namespace, oldFinetune.Status.State, newFinetune.Status.State)
					return true
				}
				return false
			},
			CreateFunc: func(createEvent event.CreateEvent) bool {
				finetune := createEvent.Object.(*finetunev1beta1.Finetune)
				r.Log.Infof("Get finetun %s/%s crate event, skip", finetune.Name, finetune.Namespace)
				return false
			},
		})).
		WithOptions(controller.Options{
			CacheSyncTimeout:        10 * time.Second,
			MaxConcurrentReconciles: 1}).
		Complete(r)
}
