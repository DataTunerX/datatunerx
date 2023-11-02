package app

import (
	"context"
	"os"

	"github.com/DataTunerX/finetune-experiment-controller/cmd/controller-manager/app/options"
	finetunecontrollers "github.com/DataTunerX/finetune-experiment-controller/controllers/finetune"
	"github.com/DataTunerX/finetune-experiment-controller/pkg/logging"
	corev1beta1 "github.com/DataTunerX/meta-server/api/core/v1beta1"
	finetunev1beta1 "github.com/DataTunerX/meta-server/api/finetune/v1beta1"
	"github.com/go-logr/zapr"
	"github.com/operator-framework/operator-lib/leader"
	"github.com/spf13/pflag"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	_ "k8s.io/client-go/plugin/pkg/client/auth"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	//+kubebuilder:scaffold:imports
)

const LockName = "finetune-experiment-controller-lock"

var (
	scheme = runtime.NewScheme()
)

func init() {
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
	utilruntime.Must(finetunev1beta1.AddToScheme(scheme))
	utilruntime.Must(corev1beta1.AddToScheme(scheme))
	//+kubebuilder:scaffold:scheme
}

func NewControllerManager() (manager.Manager, error) {
	logging.ZLogger.Info("Start building controller manager")
	opts := options.NewOptions()
	flagSet := pflag.NewFlagSet("generic", pflag.ExitOnError)
	opts.AddFlags(flagSet)
	err := flagSet.Parse(os.Args[1:])
	if err != nil {
		logging.ZLogger.Errorf("Error parsing flags: %v", err)
		return nil, err
	}
	logging.ZLogger.Info("Set logger for controller")
	ctrl.SetLogger(zapr.NewLogger(logging.ZLogger.GetLogger()))

	ctrOption := ctrl.Options{
		Scheme:                 scheme,
		MetricsBindAddress:     opts.MetricsAddr,
		Port:                   9443,
		HealthProbeBindAddress: opts.ProbeAddr,
	}

	if opts.LeaderElectLifeConfig.EnableLeaderLifeElect {
		err = leader.Become(context.TODO(), LockName)
		if err != nil {
			logging.ZLogger.Errorf("Failed to retry for leader lock: %v", err)
			return nil, err
		}
	} else {
		ctrOption.LeaderElection = false
		ctrOption.LeaderElectionID = LockName
		ctrOption.RetryPeriod = &opts.LeaderElectLeaseConfig.RetryPeriod
		ctrOption.RenewDeadline = &opts.LeaderElectLeaseConfig.RenewDeadline
		ctrOption.LeaseDuration = &opts.LeaderElectLeaseConfig.LeaseDuration
		ctrOption.LeaderElectionNamespace = "default"
	}

	mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrOption)
	if err != nil {
		logging.ZLogger.Errorf("Build controller manager failed: %v", err)
		return nil, err
	}

	if err = (&finetunecontrollers.FinetuneExperimentReconciler{
		Client: mgr.GetClient(),
		Scheme: mgr.GetScheme(),
	}).SetupWithManager(mgr); err != nil {
		logging.ZLogger.Errorf("Unable to create FinetuneExperiment controller, %v", err)
		return nil, err
	}
	if err = (&finetunecontrollers.FinetuneJobReconciler{
		Client: mgr.GetClient(),
		Scheme: mgr.GetScheme(),
	}).SetupWithManager(mgr); err != nil {
		logging.ZLogger.Errorf("Unable to create FinetuneJob controller, %v", err)
		return nil, err
	}
	//+kubebuilder:scaffold:builder

	if err := mgr.AddHealthzCheck("healthz", healthz.Ping); err != nil {
		logging.ZLogger.Errorf("Unable to set up health check: %v", err)
		return nil, err
	}
	if err := mgr.AddReadyzCheck("readyz", healthz.Ping); err != nil {
		logging.ZLogger.Errorf("Unable to set up ready check: %v", err)
		return nil, err
	}

	return mgr, nil
}
