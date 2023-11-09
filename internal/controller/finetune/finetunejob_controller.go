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
	"reflect"
	"time"

	"github.com/DataTunerX/finetune-experiment-controller/pkg/util/generate"
	"github.com/DataTunerX/finetune-experiment-controller/pkg/util/handlererr"
	corev1beta1 "github.com/DataTunerX/meta-server/api/core/v1beta1"
	extensionv1beta1 "github.com/DataTunerX/meta-server/api/extension/v1beta1"
	finetunev1beta1 "github.com/DataTunerX/meta-server/api/finetune/v1beta1"
	"github.com/DataTunerX/utility-server/logging"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/event"
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
	r.Log.Infof("Start reconcile finetuneJob: %s/%s,", req.Name, req.Namespace)
	finetuneJob := &finetunev1beta1.FinetuneJob{}
	if err := r.Get(ctx, req.NamespacedName, finetuneJob); err != nil {
		if errors.IsNotFound(err) {
			r.Log.Errorf("FinetuneJob resource not found. Ignoring since object must be deleted.")
			return handlererr.HandlerErr(nil)
		}
		r.Log.Errorf("Failed get finetuneJob: %s/%s, Err: %v", req.Name, req.Namespace, err)
		return handlererr.HandlerErr(err)
	}

	if finetuneJob.GetDeletionTimestamp() != nil {
		r.Log.Infof("Start delete finetuneJob: %s/%s", req.Name, req.Namespace)
		if controllerutil.ContainsFinalizer(finetuneJob, finetuneFinalizer) {
			// todo cleaner
			controllerutil.RemoveFinalizer(finetuneJob, finetuneFinalizer)
			if err := r.Update(ctx, finetuneJob); err != nil {
				r.Log.Errorf("Remove finalizer failed: %s/%s, Err: %v", req.Name, req.Namespace, err)
				return handlererr.HandlerErr(err)
			}
		}
		return handlererr.HandlerErr(nil)
	}
	if !controllerutil.ContainsFinalizer(finetuneJob, finetuneFinalizer) {
		controllerutil.AddFinalizer(finetuneJob, finetuneFinalizer)
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
	if err := r.Client.Get(ctx, types.NamespacedName{Name: finetuneJob.Spec.FineTune.FinetuneSpec.Hyperparameter, Namespace: req.Namespace}, hyperparameter); err != nil {
		r.Log.Errorf("Failed get hyperparameter: %s/%s, err: %v", finetuneJob.Spec.FineTune.FinetuneSpec.Hyperparameter, req.Namespace, err)
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
	// Phase III of the fine-tuning exercise.
	// Update finetunejob status.
	r.Log.Infof("update finetuneJob %s/%s status %s.", req.Name, req.Namespace, finetunev1beta1.FinetuneJobFinetune)
	finetuneJob.Status.State = finetunev1beta1.FinetuneJobFinetune
	if err := r.Client.Status().Update(ctx, finetuneJob); err != nil {
		r.Log.Errorf("Update finetune status failed: %v", err)
		return handlererr.HandlerErr(err)
	}
	// Phase IIII of the fine-tuning exercise.
	// Check finetune cr status, if finetune cr status is SUCCESSFUL, start next
	return handlererr.HandlerErr(nil)
}

// SetupWithManager sets up the controller with the Manager.
func (r *FinetuneJobReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&finetunev1beta1.FinetuneJob{}).
		WithEventFilter(predicate.Funcs{
			UpdateFunc: func(updateEvent event.UpdateEvent) bool {
				oldFinetuneJob := updateEvent.ObjectOld.(*finetunev1beta1.FinetuneJob)
				newFinetuneJob := updateEvent.ObjectNew.(*finetunev1beta1.FinetuneJob)
				if !reflect.DeepEqual(oldFinetuneJob.Spec, newFinetuneJob.Spec) || !newFinetuneJob.GetDeletionTimestamp().IsZero() {
					return true
				}
				return false
			},
			DeleteFunc: func(deleteEvent event.DeleteEvent) bool {
				return false
			},
		}).
		WithOptions(controller.Options{
			CacheSyncTimeout:        10 * time.Second,
			MaxConcurrentReconciles: 1}).
		Complete(r)
}
