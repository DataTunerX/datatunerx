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

	"github.com/DataTunerX/finetune-experiment-controller/pkg/util/handlererr"
	finetunev1beta1 "github.com/DataTunerX/meta-server/api/finetune/v1beta1"
	"github.com/DataTunerX/utility-server/logging"
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
	"sigs.k8s.io/controller-runtime/pkg/source"
)

// FinetuneExperimentReconciler reconciles a FinetuneExperiment object
type FinetuneExperimentReconciler struct {
	client.Client
	Scheme *runtime.Scheme
	Log    logging.Logger
}

//+kubebuilder:rbac:groups=finetune.datatunerx.io,resources=finetuneexperiments,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=finetune.datatunerx.io,resources=finetuneexperiments/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=finetune.datatunerx.io,resources=finetuneexperiments/finalizers,verbs=update

func (r *FinetuneExperimentReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	r.Log.Infof("Start reconcile finetuneExperiment: %s/%s,", req.Name, req.Namespace)
	finetuneExperiment := &finetunev1beta1.FinetuneExperiment{}
	if err := r.Get(ctx, req.NamespacedName, finetuneExperiment); err != nil {
		if errors.IsNotFound(err) {
			r.Log.Infof("FinetuneExperiment resource not found. Ignoring since object must be deleted.")
			return handlererr.HandlerErr(nil)
		}
		r.Log.Errorf("Failed get finetuneExperiment: %s/%s, Err: %v", req.Name, req.Namespace, err)
		return handlererr.HandlerErr(err)
	}

	if finetuneExperiment.GetDeletionTimestamp() != nil {
		if controllerutil.ContainsFinalizer(finetuneExperiment, finetunev1beta1.FinetuneGroupFinalizer) {
			// todo cleaner
			controllerutil.RemoveFinalizer(finetuneExperiment, finetunev1beta1.FinetuneGroupFinalizer)
			if err := r.Update(ctx, finetuneExperiment); err != nil {
				r.Log.Errorf("Remove finalizer failed: %s/%s, Err: %v", req.Name, req.Namespace, err)
				return handlererr.HandlerErr(err)
			}
		}
		return handlererr.HandlerErr(nil)
	}
	if !controllerutil.ContainsFinalizer(finetuneExperiment, finetunev1beta1.FinetuneGroupFinalizer) {
		controllerutil.AddFinalizer(finetuneExperiment, finetunev1beta1.FinetuneGroupFinalizer)
		err := r.Update(ctx, finetuneExperiment)
		if err != nil {
			r.Log.Errorf("Add finalizer failed: %s/%s, %v", req.Name, req.Namespace, err)
			return handlererr.HandlerErr(err)
		}
	}

	if finetuneExperiment.Spec.Pending {
		finetuneExperiment.Status.State = finetunev1beta1.FinetuneExperimentPending
		finetuneExperiment.Status.Stats = metav1.Now().Format("2006-01-02 15:04:05")
		if err := r.Client.Status().Update(ctx, finetuneExperiment); err != nil {
			r.Log.Errorf("Update fineExperiment %s/%s status failed", finetuneExperiment.Name, finetuneExperiment.Namespace)
			return handlererr.HandlerErr(err)
		}
		return handlererr.HandlerErr(nil)
	}

	for i := range finetuneExperiment.Spec.FinetuneJobs {
		finetuneJob := finetuneExperiment.Spec.FinetuneJobs[i]
		if finetuneJob.Name == "" {
			finetuneJob.Name = fmt.Sprintf("%s-%s-%d", finetuneExperiment.Name, "finetunejob", i+1)
			finetuneExperiment.Spec.FinetuneJobs[i].Name = fmt.Sprintf("%s-%s-%d", finetuneExperiment.Name, "finetunejob", i+1)
		}
		existFinetuneJob := &finetunev1beta1.FinetuneJob{}
		if err := r.Client.Get(ctx, types.NamespacedName{
			Name:      finetuneJob.Name,
			Namespace: finetuneExperiment.Namespace,
		}, existFinetuneJob); err != nil {
			if errors.IsNotFound(err) {
				finetuneJobInstance := &finetunev1beta1.FinetuneJob{}
				finetuneJobInstance.Spec = finetuneJob.Spec
				finetuneJobInstance.Name = finetuneJob.Name
				r.Log.Infof("finetuneJob Name: %s", finetuneJobInstance.Name)
				finetuneJobInstance.Namespace = finetuneExperiment.Namespace
				if err := ctrl.SetControllerReference(finetuneExperiment, finetuneJobInstance, r.Scheme); err != nil {
					r.Log.Errorf("SetControllerReference failed finetuneJob: %s/%s, owner finetuneExperiment: %s/%s, err: %v",
						finetuneJobInstance.Name, finetuneJobInstance.Namespace, finetuneExperiment.Name, finetuneExperiment.Namespace, err)
					return handlererr.HandlerErr(err)
				}
				if err := r.Client.Create(ctx, finetuneJobInstance); err != nil {
					if !errors.IsAlreadyExists(err) {
						r.Log.Errorf("Create finetuneJob %s/%s failed: %v", finetuneJobInstance.Name, finetuneJobInstance.Namespace, err)
						return handlererr.HandlerErr(err)
					}
				}
			} else {
				r.Log.Errorf("Get finetuneJob %s/%s failed: %v", finetuneJob.Name, finetuneExperiment.Namespace, err)
				return handlererr.HandlerErr(err)
			}
		}
	}
	if finetuneExperiment.Status.State == finetunev1beta1.FinetuneExperimentProcessing {
		for i := range finetuneExperiment.Spec.FinetuneJobs {
			if finetuneExperiment.Spec.FinetuneJobs[i].Name == "" {
				finetuneExperiment.Spec.FinetuneJobs[i].Name = fmt.Sprintf("%s-%s-%d", finetuneExperiment.Name, "finetunejob", i+1)
			}
			finetuneJobInstance := &finetunev1beta1.FinetuneJob{}
			if err := r.Client.Get(ctx, types.NamespacedName{Name: finetuneExperiment.Spec.FinetuneJobs[i].Name, Namespace: finetuneExperiment.Namespace}, finetuneJobInstance); err != nil {
				r.Log.Errorf("Get finetuneJob %s/%s failed, err: %v", finetuneExperiment.Spec.FinetuneJobs[i].Name, finetuneExperiment.Namespace, err)
				return handlererr.HandlerErr(err)
			}
			if finetuneJobInstance.Status.FinetuneState == "" {
				finetuneJobInstance.Status.State = finetunev1beta1.FinetuneJobInit
			}

			if finetuneExperiment.Status.JobsStatus == nil {
				finetuneExperiment.Status.JobsStatus = make([]*finetunev1beta1.FinetuneJobStatusSetting, len(finetuneExperiment.Spec.FinetuneJobs))
			}
			if finetuneExperiment.Status.JobsStatus[i] != nil {
				if !reflect.DeepEqual(finetuneExperiment.Status.JobsStatus[i].FinetuneJobStatus, finetuneJobInstance.Status) {
					finetuneExperiment.Status.JobsStatus[i] = &finetunev1beta1.FinetuneJobStatusSetting{
						Name:              finetuneJobInstance.Name,
						FinetuneJobStatus: finetuneJobInstance.Status,
					}
				}
			} else {
				finetuneExperiment.Status.JobsStatus[i] = &finetunev1beta1.FinetuneJobStatusSetting{
					Name:              finetuneJobInstance.Name,
					FinetuneJobStatus: finetuneJobInstance.Status,
				}
			}
		}
		if err := r.Client.Update(ctx, finetuneExperiment); err != nil {
			r.Log.Errorf("Update fineExperiment %s/%s failed", finetuneExperiment.Name, finetuneExperiment.Namespace)
			return handlererr.HandlerErr(err)
		}
		if err := r.Client.Status().Update(ctx, finetuneExperiment); err != nil {
			r.Log.Errorf("Update fineExperiment %s/%s status failed", finetuneExperiment.Name, finetuneExperiment.Namespace)
			return handlererr.HandlerErr(err)
		}
	}
	if finetuneExperiment.Status.State == "" {
		finetuneExperiment.Status.State = finetunev1beta1.FinetuneExperimentProcessing
		if err := r.Client.Status().Update(ctx, finetuneExperiment); err != nil {
			r.Log.Errorf("Update fineExperiment %s/%s status failed", finetuneExperiment.Name, finetuneExperiment.Namespace)
			return handlererr.HandlerErr(err)
		}
	}
	return handlererr.HandlerErr(nil)
}

// SetupWithManager sets up the controller with the Manager.
func (r *FinetuneExperimentReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&finetunev1beta1.FinetuneExperiment{}).
		Watches(&source.Kind{Type: &finetunev1beta1.FinetuneJob{}}, &handler.EnqueueRequestForOwner{
			OwnerType:    &finetunev1beta1.FinetuneExperiment{},
			IsController: true,
		}, builder.WithPredicates(predicate.Funcs{
			UpdateFunc: func(updateEvent event.UpdateEvent) bool {
				oldFinetuneJob := updateEvent.ObjectOld.(*finetunev1beta1.FinetuneJob)
				newFinetuneJob := updateEvent.ObjectNew.(*finetunev1beta1.FinetuneJob)
				if oldFinetuneJob.Status.State != newFinetuneJob.Status.State {
					r.Log.Infof("Get finetuneJob %s/%s update event oldStatus: %s, newStatus: %s", oldFinetuneJob.Namespace, oldFinetuneJob.Name, oldFinetuneJob.Status.State, newFinetuneJob.Status.State)
					return true
				}
				return false
			},
			CreateFunc: func(createEvent event.CreateEvent) bool {
				finetuneJob := createEvent.Object.(*finetunev1beta1.FinetuneJob)
				r.Log.Infof("Get finetuneJob %s/%s crate event, skip", finetuneJob.Name, finetuneJob.Namespace)
				return false
			},
		})).
		WithOptions(controller.Options{
			CacheSyncTimeout:        10 * time.Second,
			MaxConcurrentReconciles: 1}).
		Complete(r)
}
