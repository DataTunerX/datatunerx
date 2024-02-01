package app

import (
	"fmt"
	"os"

	"github.com/DataTunerX/finetune-experiment-controller/cmd/controller-manager/app/options"
	"github.com/DataTunerX/finetune-experiment-controller/internal/controller/finetune"
	"github.com/DataTunerX/finetune-experiment-controller/pkg/util"
	corev1beta1 "github.com/DataTunerX/meta-server/api/core/v1beta1"
	extensionv1beta1 "github.com/DataTunerX/meta-server/api/extension/v1beta1"
	finetunev1beta1 "github.com/DataTunerX/meta-server/api/finetune/v1beta1"
	"github.com/DataTunerX/utility-server/logging"
	"github.com/go-logr/zapr"
	"github.com/open-policy-agent/cert-controller/pkg/rotator"
	rayv1 "github.com/ray-project/kuberay/ray-operator/apis/ray/v1"
	"github.com/spf13/pflag"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/kubernetes"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	_ "k8s.io/client-go/plugin/pkg/client/auth"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
	"sigs.k8s.io/controller-runtime/pkg/webhook"
	//+kubebuilder:scaffold:imports
)

const (
	LockName       = "datatunerx-lock"
	SecretName     = "datatunerx-cert"
	CaName         = "datatunerx-ca"
	CaOrganization = "datatunerx"
	ServiceName    = "finetune-experiment"
)

var (
	scheme = runtime.NewScheme()
)

func init() {
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
	utilruntime.Must(finetunev1beta1.AddToScheme(scheme))
	utilruntime.Must(corev1beta1.AddToScheme(scheme))
	utilruntime.Must(extensionv1beta1.AddToScheme(scheme))
	utilruntime.Must(rayv1.AddToScheme(scheme))
	//+kubebuilder:scaffold:scheme
}

func NewControllerManager() (manager.Manager, error) {
	opts := options.NewOptions()
	flagSet := pflag.NewFlagSet("generic", pflag.ExitOnError)
	opts.AddFlags(flagSet)
	err := flagSet.Parse(os.Args[1:])
	if err != nil {
		logging.ZLogger.Errorf("Error parsing flags: %v", err)
		os.Exit(1)
	}
	logging.ZLogger.Info("Set logger for controller")
	ctrl.SetLogger(zapr.NewLogger(logging.ZLogger.GetLogger()))
	namespace := util.GetOperatorNamespace()
	ctrOption := ctrl.Options{
		Scheme: scheme,
		Metrics: metricsserver.Options{
			BindAddress: opts.MetricsAddr,
		},
		WebhookServer:           webhook.NewServer(webhook.Options{Port: 9443}),
		HealthProbeBindAddress:  opts.ProbeAddr,
		LeaderElection:          true,
		LeaderElectionID:        LockName,
		LeaderElectionNamespace: namespace,
	}

	mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrOption)
	if err != nil {
		logging.ZLogger.Errorf("Build controller manager failed: %v", err)
		return nil, err
	}
	setupFinished := make(chan struct{})
	if opts.EnableCertRotator {
		logging.ZLogger.Info("Setting up cert rotation")
		if err := rotator.AddRotator(mgr, &rotator.CertRotator{
			SecretKey: types.NamespacedName{
				Namespace: namespace,
				Name:      SecretName,
			},
			CAName:         CaName,
			CAOrganization: CaOrganization,
			CertDir:        "/tmp/k8s-webhook-server/serving-certs",
			DNSName:        fmt.Sprintf("%s.%s.svc", ServiceName, namespace),
			IsReady:        setupFinished,
			Webhooks: []rotator.WebhookInfo{
				{
					Name: namespace + "-validating-webhook-configuration",
					Type: rotator.Validating,
				},
				{
					Name: namespace + "-mutating-webhook-configuration",
					Type: rotator.Mutating,
				},
			},
		}); err != nil {
			logging.ZLogger.Errorf("Unable to set up cert rotation, %v", err)
			os.Exit(1)
		}
	} else {
		close(setupFinished)
	}
	go func() {
		<-setupFinished
		if err := (&finetunev1beta1.FinetuneJob{}).SetupWebhookWithManager(mgr); err != nil {
			logging.ZLogger.Errorf("Unable to create webhook, %v", err)
			os.Exit(1)

		}
		if err := (&finetunev1beta1.FinetuneExperiment{}).SetupWebhookWithManager(mgr); err != nil {
			logging.ZLogger.Errorf("Unable to create webhook, %v", err)
			os.Exit(1)
		}
		if err := (&corev1beta1.LLM{}).SetupWebhookWithManager(mgr); err != nil {
			logging.ZLogger.Errorf("Unable to create webhook, %v", err)
			os.Exit(1)
		}
		if err := (&corev1beta1.Hyperparameter{}).SetupWebhookWithManager(mgr); err != nil {
			logging.ZLogger.Errorf("Unable to create webhook, %v", err)
			os.Exit(1)
		}
		if err := (&extensionv1beta1.Dataset{}).SetupWebhookWithManager(mgr); err != nil {
			logging.ZLogger.Errorf("Unable to create webhook, %v", err)
			os.Exit(1)
		}
	}()

	if err = (&finetune.FinetuneExperimentReconciler{
		Client: mgr.GetClient(),
		Scheme: mgr.GetScheme(),
		Log:    logging.ZLogger,
	}).SetupWithManager(mgr); err != nil {
		logging.ZLogger.Errorf("Unable to create FinetuneExperiment controller, %v", err)
		return nil, err
	}
	if err = (&finetune.FinetuneJobReconciler{
		Client: mgr.GetClient(),
		Scheme: mgr.GetScheme(),
		Log:    logging.ZLogger,
	}).SetupWithManager(mgr); err != nil {
		logging.ZLogger.Errorf("Unable to create FinetuneJob controller, %v", err)
		return nil, err
	}
	if err = (&finetune.FinetuneReconciler{
		Log:       logging.ZLogger,
		Client:    mgr.GetClient(),
		Scheme:    mgr.GetScheme(),
		Clientset: kubernetes.NewForConfigOrDie(ctrl.GetConfigOrDie()),
		Config:    ctrl.GetConfigOrDie(),
	}).SetupWithManager(mgr); err != nil {
		logging.ZLogger.Errorf("Unable to create Finetune controller, %v", err)
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
