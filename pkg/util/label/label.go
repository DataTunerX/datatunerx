package label

const (
	LabelDefaultComponent   = "finetunejob"
	LabelDefaultKey         = "finetune.datatunerx.io/component"
	LabelFinetuneBindingKey = "finetune.datatunerx.io/finetunebinding"
)

func GetBaseLabel() map[string]string {
	return map[string]string{
		LabelDefaultKey: LabelDefaultComponent,
	}
}

func MergeLabel(baseLabel map[string]string, customLabel map[string]string) map[string]string {
	for k, v := range customLabel {
		if _, exists := baseLabel[k]; !exists {
			baseLabel[k] = v
		}
	}
	return baseLabel
}
