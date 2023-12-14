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
	"github.com/DataTunerX/finetune-experiment-controller/pkg/domain/valueobject"
	"github.com/DataTunerX/finetune-experiment-controller/pkg/util"
	"github.com/DataTunerX/finetune-experiment-controller/pkg/util/generate"
	"github.com/DataTunerX/finetune-experiment-controller/pkg/util/handlererr"
	corev1beta1 "github.com/DataTunerX/meta-server/api/core/v1beta1"
	extensionv1beta1 "github.com/DataTunerX/meta-server/api/extension/v1beta1"
	finetunev1beta1 "github.com/DataTunerX/meta-server/api/finetune/v1beta1"
	"github.com/DataTunerX/utility-server/logging"
	rayv1 "github.com/ray-project/kuberay/ray-operator/apis/ray/v1"
	batchv1 "k8s.io/api/batch/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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
	r.Log.Infof("Start reconcile finetuneJob: %s/%s", req.Namespace, req.Name)
	finetuneJob := &finetunev1beta1.FinetuneJob{}
	if err := r.Get(ctx, req.NamespacedName, finetuneJob); err != nil {
		if errors.IsNotFound(err) {
			r.Log.Infof("FinetuneJob %s resource not found. Ignoring since object must be deleted.", req.NamespacedName)
			return handlererr.HandlerErr(nil)
		}
		r.Log.Errorf("Failed get finetuneJob: %s/%s, Err: %v", req.Namespace, req.Name, err)
		return handlererr.HandlerErr(err)
	}
	if finetuneJob.GetDeletionTimestamp() != nil {
		r.Log.Infof("Delete finetuneJob: %s/%s", finetuneJob.Namespace, finetuneJob.Name)
		if controllerutil.ContainsFinalizer(finetuneJob, finetunev1beta1.FinetuneGroupFinalizer) {
			// todo cleaner
			controllerutil.RemoveFinalizer(finetuneJob, finetunev1beta1.FinetuneGroupFinalizer)
			if err := r.Update(ctx, finetuneJob); err != nil {
				r.Log.Errorf("Remove finalizer failed: %s/%s, Err: %v", finetuneJob.Namespace, finetuneJob.Name, err)
				return handlererr.HandlerErr(err)
			}
		}
		return handlererr.HandlerErr(nil)
	}
	if !controllerutil.ContainsFinalizer(finetuneJob, finetunev1beta1.FinetuneGroupFinalizer) {
		controllerutil.AddFinalizer(finetuneJob, finetunev1beta1.FinetuneGroupFinalizer)
		err := r.Update(ctx, finetuneJob)
		if err != nil {
			r.Log.Errorf("Add finalizer failed: %s/%s, %v", finetuneJob.Namespace, finetuneJob.Name, err)
			return handlererr.HandlerErr(err)
		}
	}

	if finetuneJob.Status.State == "" {
		finetuneJob.Status.State = finetunev1beta1.FinetuneJobInit
		if err := r.Client.Status().Update(ctx, finetuneJob); err != nil {
			r.Log.Errorf("Update finetuneJob %s/%s status failed: %v", finetuneJob.Namespace, finetuneJob.Name, err)
			return handlererr.HandlerErr(err)
		}
	}

	if err := r.reconcilePreCondition(ctx, finetuneJob); err != nil {
		return handlererr.HandlerErr(err)
	}

	existFinetune, err := r.reconcileFinetuneSend(ctx, finetuneJob)
	if err != nil {
		return handlererr.HandlerErr(err)
	}

	if err := r.reconcileByFinetuneStatus(ctx, existFinetune, finetuneJob); err != nil {
		return handlererr.HandlerErr(err)
	}

	if err := r.reconcileByJobStatus(ctx, finetuneJob, existFinetune); err != nil {
		return handlererr.HandlerErr(err)
	}

	if err := r.reconcileByRayServiceStatus(ctx, finetuneJob); err != nil {
		return handlererr.HandlerErr(err)
	}

	if err := r.reconcileByScoringStatus(ctx, finetuneJob); err != nil {
		return handlererr.HandlerErr(err)
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
		Watches(&finetunev1beta1.Finetune{},
			handler.EnqueueRequestForOwner(mgr.GetScheme(), mgr.GetRESTMapper(), &finetunev1beta1.FinetuneJob{}, handler.OnlyControllerOwner()),
			builder.WithPredicates(predicate.Funcs{
				UpdateFunc: func(updateEvent event.UpdateEvent) bool {
					oldFinetune := updateEvent.ObjectOld.(*finetunev1beta1.Finetune)
					newFinetune := updateEvent.ObjectNew.(*finetunev1beta1.Finetune)
					if oldFinetune.Status.State != newFinetune.Status.State {
						r.Log.Infof("Get finetun %s/%s update event oldStatus: %s, newStatus: %s", oldFinetune.Name, oldFinetune.Namespace, oldFinetune.Status.State, newFinetune.Status.State)
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
		Watches(&batchv1.Job{},
			handler.EnqueueRequestForOwner(mgr.GetScheme(), mgr.GetRESTMapper(), &finetunev1beta1.FinetuneJob{}, handler.OnlyControllerOwner()),
			builder.WithPredicates(predicate.NewPredicateFuncs(func(object client.Object) bool {
				job := object.(*batchv1.Job)
				if job.Status.CompletionTime != nil {
					return true
				}
				return false
			}))).
		Watches(&rayv1.RayService{},
			handler.EnqueueRequestForOwner(mgr.GetScheme(), mgr.GetRESTMapper(), &finetunev1beta1.FinetuneJob{}, handler.OnlyControllerOwner()),
			builder.WithPredicates(predicate.NewPredicateFuncs(func(object client.Object) bool {
				rayService := object.(*rayv1.RayService)
				if rayService.Status.ServiceStatus == rayv1.Running {
					return true
				}
				return false
			}))).
		Watches(&extensionv1beta1.Scoring{},
			handler.EnqueueRequestForOwner(mgr.GetScheme(), mgr.GetRESTMapper(), &finetunev1beta1.FinetuneJob{}, handler.OnlyControllerOwner()),
			builder.WithPredicates(predicate.NewPredicateFuncs(func(object client.Object) bool {
				scoring := object.(*extensionv1beta1.Scoring)
				if scoring.Status.Score != nil {
					return true
				}
				return false
			}))).
		WithOptions(controller.Options{
			CacheSyncTimeout:        10 * time.Second,
			MaxConcurrentReconciles: 1}).
		Complete(r)
}

func (r *FinetuneJobReconciler) reconcilePreCondition(ctx context.Context, finetuneJob *finetunev1beta1.FinetuneJob) error {
	preCondition := make(map[string]client.Object, 3)
	preCondition[finetuneJob.Spec.FineTune.FinetuneSpec.LLM] = &corev1beta1.LLM{}
	preCondition[finetuneJob.Spec.FineTune.FinetuneSpec.Hyperparameter.HyperparameterRef] = &corev1beta1.Hyperparameter{}
	preCondition[finetuneJob.Spec.FineTune.FinetuneSpec.Dataset] = &extensionv1beta1.Dataset{}
	for name, obj := range preCondition {
		if err := r.Get(ctx, types.NamespacedName{Name: name, Namespace: finetuneJob.Namespace}, obj); err != nil {
			r.Log.Errorf("Get %s: %s/%s failed, err: %v", obj.GetObjectKind(), finetuneJob.Namespace, name, err)
			return err
		}
	}
	return nil
}

func (r *FinetuneJobReconciler) reconcileFinetuneSend(ctx context.Context, finetuneJob *finetunev1beta1.FinetuneJob) (*finetunev1beta1.Finetune, error) {

	specFinetuneInstance := generate.GenerateFinetune(finetuneJob)
	existFinetuneInstance := &finetunev1beta1.Finetune{}
	if err := r.Get(ctx, types.NamespacedName{
		Name:      specFinetuneInstance.Name,
		Namespace: specFinetuneInstance.Namespace,
	}, existFinetuneInstance); err != nil {
		if errors.IsNotFound(err) {
			r.Log.Infof("Starting to send down finetune: %s/%s.", specFinetuneInstance.Namespace, specFinetuneInstance.Name)
			if err := ctrl.SetControllerReference(finetuneJob, specFinetuneInstance, r.Scheme); err != nil {
				r.Log.Errorf("For %s/%s set owner %s/%s failed: %v", specFinetuneInstance.Namespace, specFinetuneInstance.Name, finetuneJob.Namespace, finetuneJob.Name, err)
				return nil, err
			}
			if err := r.Client.Create(ctx, specFinetuneInstance); err != nil {
				if !errors.IsAlreadyExists(err) {
					r.Log.Errorf("Create finetune %s/%s failed: %v", specFinetuneInstance.Namespace, specFinetuneInstance.Name, err)
					return nil, err
				}
			}
			return nil, valueobject.ErrRecalibrate
		}
	}
	return existFinetuneInstance, nil
}

func (r *FinetuneJobReconciler) reconcileByFinetuneStatus(ctx context.Context, finetuneInstance *finetunev1beta1.Finetune, finetuneJobInstance *finetunev1beta1.FinetuneJob) error {

	if finetuneInstance.Status.State == finetunev1beta1.FinetuneInit || finetuneInstance.Status.State == finetunev1beta1.FinetuneRunning {
		r.Log.Infof("Update finetuneJob %s/%s status %s.", finetuneJobInstance.Namespace, finetuneJobInstance.Name, finetunev1beta1.FinetuneJobFinetune)
		finetuneJobInstance.Status.State = finetunev1beta1.FinetuneJobFinetune
		finetuneJobInstance.Status.FinetuneStatus = &finetuneInstance.Status
		if err := r.Client.Status().Update(ctx, finetuneJobInstance); err != nil {
			r.Log.Errorf("Update finetuneJob %s/%s status failed: %v", finetuneJobInstance.Namespace, finetuneJobInstance.Name, err)
			return err
		}
	}

	if finetuneInstance.Status.State == finetunev1beta1.FinetuneSuccessful && finetuneJobInstance.Status.State != finetunev1beta1.FinetuneJobBuildImage {
		if finetuneInstance.Status.LLMCheckpoint == nil {
			r.Log.Infof("Finetune %s/%s status not found llmCheckpointRef", finetuneInstance.Namespace, finetuneInstance.Name)
			return fmt.Errorf("Finetune %s/%s status not set llmCheckpointRef", finetuneInstance.Namespace, finetuneInstance.Name)
		}

		llmCheckpoint := &corev1beta1.LLMCheckpoint{}
		if err := r.Get(ctx, types.NamespacedName{Name: finetuneInstance.Status.LLMCheckpoint.LLMCheckpointRef, Namespace: finetuneJobInstance.Namespace}, llmCheckpoint); err != nil {
			r.Log.Errorf("Get llmCheckpoint %s/%s failed, err: %v", finetuneJobInstance.Namespace, finetuneInstance.Status.LLMCheckpoint, err)
			return err
		}
		// build llmCheckpoint image server. job

		imageName := fmt.Sprintf("ray271-llama2-7b-finetune-checkpoint-%s", finetuneJobInstance.Name)
		imageTag := fmt.Sprintf("%s", time.Now().Format("20060102"))
		checkPointFilePath := finetuneInstance.Status.LLMCheckpoint.CheckpointPath
		checkPointFilePath = util.RemoveBucketName(checkPointFilePath, config.GetS3Bucket())
		buildImageJob := generate.GenerateBuildImageJob(checkPointFilePath, imageName, imageTag, finetuneJobInstance)
		if err := ctrl.SetControllerReference(finetuneJobInstance, buildImageJob, r.Scheme); err != nil {
			r.Log.Errorf("Set owner failed: %v", err)
			return err
		}
		if err := r.Client.Get(ctx, types.NamespacedName{Name: buildImageJob.Name, Namespace: buildImageJob.Namespace}, buildImageJob); err != nil {
			if errors.IsNotFound(err) {
				if err := r.Client.Create(ctx, buildImageJob); err != nil {
					r.Log.Errorf("Create job %s/%s failed, err: %v", buildImageJob.Name, buildImageJob.Namespace, err)
					return err
				}
			}
		}

		llmCheckpoint.Spec.CheckpointImage = &corev1beta1.CheckpointImage{}
		llmImage := fmt.Sprintf("%s/%s/%s:%s", config.GetRegistryUrl(), config.GetRepositoryName(), imageName, imageTag)
		llmCheckpoint.Spec.CheckpointImage.Name = &llmImage
		llmCheckpoint.Spec.CheckpointImage.CheckPointPath = fmt.Sprintf("/checkpoint/%s", checkPointFilePath)
		llmCheckpoint.Spec.CheckpointImage.LLMPath = llmCheckpoint.Spec.Image.Path
		if err := r.Client.Update(ctx, llmCheckpoint); err != nil {
			r.Log.Errorf("Update llmCheckpoint %s/%s failed: %v", llmCheckpoint.Namespace, llmCheckpoint.Name, err)
			return err
		}

		finetuneJobInstance.Status.State = finetunev1beta1.FinetuneJobBuildImage
		finetuneJobInstance.Status.FinetuneStatus = &finetuneInstance.Status
		if err := r.Client.Status().Update(ctx, finetuneJobInstance); err != nil {
			r.Log.Errorf("Update finetuneJob %s/%s status failed: %v", finetuneJobInstance.Namespace, finetuneInstance.Name, err)
			return err
		}
	}

	if finetuneInstance.Status.State == finetunev1beta1.FinetuneFailed {
		finetuneJobInstance.Status.State = finetunev1beta1.FinetuneJobFailed
		finetuneJobInstance.Status.FinetuneStatus = &finetuneInstance.Status
		if err := r.Client.Status().Update(ctx, finetuneJobInstance); err != nil {
			r.Log.Errorf("Update finetuneJob %s/%s status failed: %v", finetuneJobInstance.Namespace, finetuneInstance.Name, err)
			return err
		}
	}
	return nil
}

func (r *FinetuneJobReconciler) reconcileByJobStatus(ctx context.Context, finetuneJob *finetunev1beta1.FinetuneJob, finetune *finetunev1beta1.Finetune) error {

	jobName := fmt.Sprintf("%s-buildimage", finetuneJob.Name)
	buildImageJob := &batchv1.Job{}
	if err := r.Get(ctx, types.NamespacedName{Namespace: finetuneJob.Namespace, Name: jobName}, buildImageJob); err != nil {
		if errors.IsNotFound(err) {
			r.Log.Infof("Job %s/%s not found, waiting", finetuneJob.Namespace, jobName)
			return valueobject.ErrRecalibrate
		}
		return err
	}

	if buildImageJob.Status.CompletionTime != nil && finetuneJob.Status.State != finetunev1beta1.FinetuneJobServe {
		llmCheckpoint := &corev1beta1.LLMCheckpoint{}
		if err := r.Get(ctx, types.NamespacedName{Name: finetune.Status.LLMCheckpoint.LLMCheckpointRef, Namespace: finetuneJob.Namespace}, llmCheckpoint); err != nil {
			r.Log.Errorf("Get llmCheckpoint %s/%s failed, err: %v", finetuneJob.Namespace, finetune.Status.LLMCheckpoint.LLMCheckpointRef, err)
			return err
		}
		r.Log.Infof("Build image success, start update llmCheckpoint %s/%s", llmCheckpoint.Namespace, llmCheckpoint.Name)
		// todo(tigerK) update llmCheckpoint spec.checkpointimage
		r.Log.Infof("Update llmCheckpoint status successful, start send serve")
		rayServiceName := fmt.Sprintf("%s", finetuneJob.Name)
		importPath := fmt.Sprintf("%s.deployment", "inference")
		runtimeEnv := "working_dir: file:///home/inference/inference.zip"
		deploymentName := "LlamaDeployment"
		rayService := generate.GenerateRayService(rayServiceName,
			finetuneJob.Namespace, importPath, runtimeEnv, deploymentName,
			int32(1), float64(1), finetuneJob, llmCheckpoint)
		if err := ctrl.SetControllerReference(finetuneJob, rayService, r.Scheme); err != nil {
			r.Log.Errorf("Set owner failed: %v", err)
			return err
		}

		if err := r.Client.Get(ctx, types.NamespacedName{Name: rayServiceName, Namespace: finetuneJob.Namespace}, rayService); err != nil {
			if errors.IsNotFound(err) {
				if err := r.Create(ctx, rayService); err != nil {
					r.Log.Errorf("Create rayService %s/%s failed: %v", rayServiceName, finetuneJob.Namespace, err)
					return err
				}
			}
		}

		finetuneJob.Status.State = finetunev1beta1.FinetuneJobServe
		finetuneJob.Status.FinetuneStatus = &finetune.Status
		finetuneJob.Status.Result = &finetunev1beta1.FinetuneJobResult{
			ModelExportResult: true,
			Image:             *llmCheckpoint.Spec.CheckpointImage.Name,
		}
		if err := r.Client.Status().Update(ctx, finetuneJob); err != nil {
			r.Log.Errorf("Update finetuneJob status failed: %v", err)
			return err
		}
	}
	return nil
}

func (r *FinetuneJobReconciler) reconcileByRayServiceStatus(ctx context.Context, finetuneJob *finetunev1beta1.FinetuneJob) error {
	rayServiceName := fmt.Sprintf("%s", finetuneJob.Name)
	rayService := &rayv1.RayService{}
	if err := r.Get(ctx, types.NamespacedName{
		Name:      rayServiceName,
		Namespace: finetuneJob.Namespace,
	}, rayService); err != nil {
		r.Log.Errorf("Get rayService %s/%s failed: %v", finetuneJob.Namespace, rayServiceName, err)
		return err
	}
	if finetuneJob.Status.State == finetunev1beta1.FinetuneJobServe && rayService.Status.ServiceStatus == rayv1.Running {
		if rayService.Status.ActiveServiceStatus.Applications["default"].Deployments["LlamaDeployment"].Status == "HEALTHY" {
			// todo(tigerK) no time for optimisation
			//serveNodePort := rayService.Status.ActiveServiceStatus.RayClusterStatus.Endpoints["serve"]
			//dashboardNodePort := rayService.Status.ActiveServiceStatus.RayClusterStatus.Endpoints["dashboard"]
			finetuneJob.Status.Result.Serve = fmt.Sprintf("%s.%s.svc:%s", finetuneJob.Name, finetuneJob.Namespace, "8000")
			finetuneJob.Status.Result.Dashboard = fmt.Sprintf("%s.%s.svc:%s", finetuneJob.Name, finetuneJob.Namespace, "8080")
		} else {
			return valueobject.ErrRecalibrate
		}
		infrencePath := fmt.Sprintf("http://%s/inference", finetuneJob.Status.Result.Serve)
		if err := r.Client.Status().Update(ctx, finetuneJob); err != nil {
			r.Log.Errorf("Update finetuneJob status failed: %v", err)
			return err
		}
		scoringName := fmt.Sprintf("%s-scoring", finetuneJob.Name)
		if finetuneJob.Spec.ScoringConfig.Name == "" {
			scoring := generate.GenerateBuiltInScoring(scoringName, finetuneJob.Namespace, infrencePath)
			if err := ctrl.SetControllerReference(finetuneJob, scoring, r.Scheme); err != nil {
				r.Log.Errorf("Set owner failed: %v", err)
				return err
			}
			if err := r.Create(ctx, scoring); err != nil {
				if !errors.IsAlreadyExists(err) {
					r.Log.Errorf("Create scoring %s/%s failed: %v", scoringName, finetuneJob.Namespace, err)
					return err
				}
			}
			return nil
		}
		scoring := generate.GeneratePluginScoring(scoringName, finetuneJob.Namespace, finetuneJob.Spec.ScoringConfig.Name, finetuneJob.Spec.ScoringConfig.Parameters, infrencePath)
		if err := ctrl.SetControllerReference(finetuneJob, scoring, r.Scheme); err != nil {
			r.Log.Errorf("Set owner failed: %v", err)
			return err
		}
		if err := r.Create(ctx, scoring); err != nil {
			if !errors.IsAlreadyExists(err) {
				r.Log.Errorf("Create scoring %s/%s failed: %v", scoringName, finetuneJob.Namespace, err)
				return err
			}
		}
	}
	return nil
}

func (r *FinetuneJobReconciler) reconcileByScoringStatus(ctx context.Context, finetuneJob *finetunev1beta1.FinetuneJob) error {

	scoringName := fmt.Sprintf("%s-scoring", finetuneJob.Name)
	scoring := &extensionv1beta1.Scoring{}
	if err := r.Get(ctx, types.NamespacedName{
		Name:      scoringName,
		Namespace: finetuneJob.Namespace,
	}, scoring); err != nil {
		if errors.IsNotFound(err) {
			r.Log.Infof("Scoring %s/%s not found, err: %v", scoringName, finetuneJob.Namespace, err)
			return valueobject.ErrRecalibrate
		}
		r.Log.Errorf("Get scoring %s/%s failed: %v", scoringName, finetuneJob.Namespace, err)
		return err
	}

	// todo(tigerK) get scoring result, update finetuneJob status
	if scoring.Status.Score != nil {
		finetuneJob.Status.State = finetunev1beta1.FinetuneJobSuccessful
		finetuneJob.Status.Result.Score = *scoring.Status.Score
		finetuneJob.Status.Stats = metav1.Now().Format("2006-01-02 15:04:05")
		if err := r.Client.Status().Update(ctx, finetuneJob); err != nil {
			r.Log.Errorf("Update finetuneJob status failed: %v", err)
			return err
		}
		rayServiceName := fmt.Sprintf("%s", finetuneJob.Name)
		rayService := &rayv1.RayService{}
		if err := r.Get(ctx, types.NamespacedName{
			Name:      rayServiceName,
			Namespace: finetuneJob.Namespace,
		}, rayService); err != nil {
			if errors.IsNotFound(err) {
				return nil
			}
			r.Log.Errorf("Get rayService %s/%s failed: %v", finetuneJob.Namespace, rayServiceName, err)
			return err
		}
		if err := r.Delete(ctx, rayService); err != nil {
			r.Log.Errorf("Delete rayService %s/%s failed: %v", finetuneJob.Namespace, rayServiceName, err)
			return err
		}
	}
	return nil
}
