package valueobject

import "errors"

var (
	ErrRecalibrate = errors.New("waiting for dependent resources")
)
