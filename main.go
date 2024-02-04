/*
Copyright 2023.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package main

import (
	"os"

	"github.com/DataTunerX/datatunerx/cmd/controller-manager/app"
	"github.com/DataTunerX/datatunerx/pkg/config"
	"github.com/DataTunerX/utility-server/logging"
	ctrl "sigs.k8s.io/controller-runtime"
)

func main() {
	logging.NewZapLogger(config.GetLevel())
	controllerManager, err := app.NewControllerManager()
	if err != nil {
		os.Exit(1)
	}
	logging.ZLogger.Infof("Start controller manager")
	if err := controllerManager.Start(ctrl.SetupSignalHandler()); err != nil {
		logging.ZLogger.Errorf("Problem running manager: %v", err)
		os.Exit(1)
	}
}
