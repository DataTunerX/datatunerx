package handlererr

import (
	"time"

	ctrl "sigs.k8s.io/controller-runtime"
)

func HandlerErr(err error) (ctrl.Result, error) {
	if err != nil {
		return ctrl.Result{RequeueAfter: 20 * time.Second}, err
	}
	return ctrl.Result{}, nil
}
