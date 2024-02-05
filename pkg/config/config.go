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
	config.BindEnv("mountPath", "MOUNT_PATH")
	config.BindEnv("baseImage", "BASE_IMAGE")
	config.BindEnv("llmUrl", "LLM_URL")
	config.BindEnv("metricsExportAddress", "METRICS_EXPORT_ADDRESS")
	config.BindEnv("storagePath", "STORAGE_PATH")
	config.SetDefault("llmUrl", "/tmp/llama2-7b/")
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

func GetBaseImage() string {
	return config.GetString("baseImage")
}

func GetPassword() string {
	return config.GetString("password")
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

func GetLLMUrl() string {
	return config.GetString("llmUrl")
}

func GetStoragePath() string {
	return config.GetString("storagePath")
}

func GetMetricsExportAddress() string {
	return config.GetString("metricsExportAddress")
}
