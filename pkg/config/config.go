package config

import "github.com/spf13/viper"

var config *viper.Viper

func init() {
	config = viper.New()
	config.AutomaticEnv()
	config.BindEnv("level", "LOG_LEVEL")
	config.SetDefault("level", "debug")
	config.BindEnv("endpoint", "S3_ENDPOINT")
	config.BindEnv("accessKey", "S3_ACCESSKEYID")
	config.BindEnv("secretkey", "S3_SECRETACCESSKEY")
	config.BindEnv("bucket", "S3_BUCKET")
	config.BindEnv("secure", "S3_SECURE")
	config.BindEnv("registryUrl", "REGISTRY_URL")
	config.BindEnv("repositoryName", "REPOSITORY_NAME")
	config.BindEnv("userName", "USERNAME")
	config.BindEnv("password", "PASSWORD")
	config.BindEnv("imageName", "IMAGE_NAME")
	config.BindEnv("imageTag", "IMAGE_TAG")
	config.BindEnv("mountPath", "MOUNT_PATH")
}

func GetS3Endpoint() string {
	return config.GetString("endpoint")
}

func GetS3AccesskeyId() string {
	return config.GetString("accessKey")
}

func GetS3ESecretAccessKey() string {
	return config.GetString("secretkey")
}

func GetS3Bucket() string {
	return config.GetString("bucket")
}

func GetSecure() string {
	return config.GetString("secure")
}

func GetLevel() string {
	return config.GetString("level")
}

func GetUserName() string {
	return config.GetString("userName")
}

func GetPassword() string {
	return config.GetString("password")
}

func GetImageName() string {
	return config.GetString("imageName")
}

func GetImageTag() string {
	return config.GetString("imageTag")
}

func GetRegistryUrl() string {
	return config.GetString("registryUrl")
}

func GetRepositoryName() string {
	return config.GetString("repositoryName")
}

func GetMountPath() string {
	return config.GetString("mountPath")
}
