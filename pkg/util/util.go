package util

import "strings"

func RemoveBucketName(path, bucketName string) string {
	parts := strings.Split(path, "/")
	if len(parts) > 0 && parts[0] == bucketName {
		return strings.Join(parts[1:], "/")
	}
	return path
}

func GenerateName() {

}
