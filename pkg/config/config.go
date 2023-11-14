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
	config.BindEnv("filePath", "S3_FILEPATH")
	config.BindEnv("secure", "S3_SECURE")
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

func GetS3FilePath() string {
	return config.GetString("filePath")
}

func GetSecure() string {
	return config.GetString("secure")
}

func GetLevel() string {
	return config.GetString("level")
}
