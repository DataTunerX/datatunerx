package handlererr

import (
	"errors"
	"time"

	"github.com/DataTunerX/finetune-experiment-controller/pkg/domain/valueobject"
	ctrl "sigs.k8s.io/controller-runtime"
)

func HandlerErr(err error) (ctrl.Result, error) {
	if err != nil {
		if errors.Is(err, valueobject.ErrRecalibrate) {
			return ctrl.Result{RequeueAfter: 10 * time.Second}, nil
		}
		return ctrl.Result{RequeueAfter: 30 * time.Second}, err
	}
	return ctrl.Result{}, nil
}
