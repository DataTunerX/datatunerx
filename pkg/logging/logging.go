package logging

import (
	"github.com/DataTunerX/finetune-experiment-controller/pkg/config"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

type Logger interface {
	Debug(msg string, fields ...zap.Field)
	Debugf(format string, args ...interface{})
	Info(msg string, fields ...zap.Field)
	Infof(format string, args ...interface{})
	Error(msg string, fields ...zap.Field)
	Errorf(format string, args ...interface{})
	Warn(msg string, fields ...zap.Field)
	Warnf(format string, args ...interface{})
	GetLogger() *zap.Logger
}

var ZLogger Logger

type ZapLogger struct {
	logger *zap.Logger
	sugar  *zap.SugaredLogger
}

func NewZapLogger(logLevel string) *ZapLogger {
	var zapConfig zap.Config
	var level zapcore.Level
	if err := level.UnmarshalText([]byte(logLevel)); err != nil {
		panic(err)
	}

	zapConfig.EncoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder
	zapConfig.Level = zap.NewAtomicLevelAt(level)

	switch level {
	case zap.DebugLevel:
		zapConfig = zap.NewDevelopmentConfig()
		zapConfig.EncoderConfig.EncodeLevel = zapcore.CapitalColorLevelEncoder
	default:
		zapConfig = zap.NewProductionConfig()
		zapConfig.EncoderConfig.EncodeLevel = zapcore.CapitalLevelEncoder
	}

	logger, err := zapConfig.Build(zap.AddCaller(), zap.AddCallerSkip(1))
	if err != nil {
		panic(err)
	}

	return &ZapLogger{
		logger: logger,
		sugar:  logger.Sugar(),
	}
}

func (l *ZapLogger) Debug(msg string, fields ...zap.Field) {
	l.logger.Debug(msg, fields...)
}

func (l *ZapLogger) Debugf(format string, args ...interface{}) {
	l.sugar.Debugf(format, args...)
}

func (l *ZapLogger) Info(msg string, fields ...zap.Field) {
	l.logger.Info(msg, fields...)
}

func (l *ZapLogger) Infof(format string, args ...interface{}) {
	l.sugar.Infof(format, args...)
}

func (l *ZapLogger) Error(msg string, fields ...zap.Field) {
	l.logger.Error(msg, fields...)
}

func (l *ZapLogger) Errorf(format string, args ...interface{}) {
	l.sugar.Errorf(format, args...)
}

func (l *ZapLogger) Warn(msg string, fields ...zap.Field) {
	l.logger.Warn(msg, fields...)
}

func (l *ZapLogger) Warnf(format string, args ...interface{}) {
	l.sugar.Warnf(format, args...)
}
func (l *ZapLogger) GetLogger() *zap.Logger {
	return l.logger
}

func init() {
	ZLogger = NewZapLogger(config.GetLevel())
}
