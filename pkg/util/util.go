package util

import (
	"io/ioutil"
	"os"
	"strconv"
	"strings"

	"github.com/DataTunerX/utility-server/logging"
)

func RemoveBucketName(path, bucketName string) string {
	parts := strings.Split(path, "/")
	if len(parts) > 0 && parts[0] == bucketName {
		return strings.Join(parts[1:], "/")
	}
	return path
}

func GenerateName() {

}

func ParseScore(s string) int {
	score, err := strconv.Atoi(s)
	if err != nil {
		return 0
	}
	return score
}

func GetOperatorNamespace() string {
	nsBytes, err := ioutil.ReadFile("/var/run/secrets/kubernetes.io/serviceaccount/namespace")
	if err != nil {
		logging.ZLogger.Errorf("unable to read file, %v", err)
		if os.IsNotExist(err) {
			return "datatunerx-dev"
		}
	}
	ns := strings.TrimSpace(string(nsBytes))
	return ns
}
