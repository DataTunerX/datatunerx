package options

import (
	"time"

	"github.com/spf13/pflag"
)

const (
	defaultLeaseDuration = 10 * time.Second
	defaultRenewDeadline = 10 * time.Second
	defaultRetryPeriod   = 10 * time.Second
	defaultMetricsAddr   = ":8080"
	defaultProbeAddr     = ":8081"
)

type Options struct {
	LeaderElectLeaseConfig LeaderElectLeaseConfig
	LeaderElectLifeConfig  LeaderElectLifeConfig
	MetricsAddr            string
	ProbeAddr              string
}

type LeaderElectLeaseConfig struct {
	LeaseDuration time.Duration
	RenewDeadline time.Duration
	RetryPeriod   time.Duration
}

type LeaderElectLifeConfig struct {
	EnableLeaderLifeElect bool
}

func NewOptions() *Options {
	return &Options{
		LeaderElectLifeConfig:  LeaderElectLifeConfig{},
		LeaderElectLeaseConfig: LeaderElectLeaseConfig{},
	}
}

func (o *Options) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}
	fs.StringVar(&o.MetricsAddr, "metrics-bind-address", defaultMetricsAddr, "The address the metric endpoint binds to.")
	fs.StringVar(&o.ProbeAddr, "health-probe-bind-address", defaultProbeAddr, "The address the probe endpoint binds to.")
	fs.BoolVar(&o.LeaderElectLifeConfig.EnableLeaderLifeElect, "enable-leader-life", false, "Enable or disable leader election life.")
	fs.DurationVar(&o.LeaderElectLeaseConfig.LeaseDuration, "lease-duration", defaultLeaseDuration, "The duration that non-leader candidates will wait after observing a leadership renewal until attempting to acquire leadership of a led but unrenewed group.")
	fs.DurationVar(&o.LeaderElectLeaseConfig.RenewDeadline, "renew-deadline", defaultRenewDeadline, "Duration the clients should wait between attempting to renew the lease of the lock.")
	fs.DurationVar(&o.LeaderElectLeaseConfig.RetryPeriod, "retry-period", defaultRetryPeriod, "The time duration for the client to wait between attempts of acquiring a lock.")
}
