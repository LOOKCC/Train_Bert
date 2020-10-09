from mmsdk import mmdatasdk as md
import os


def load_dataset():
	visual_field = 'CMU_MOSI_Visual_Facet_41'
	acoustic_field = 'CMU_MOSI_COVAREP'
	text_field = 'CMU_MOSI_TimestampedWordVectors'
	label_field = 'CMU_MOSI_Opinion_Labels'

	features = [
		text_field,
		visual_field,
		acoustic_field
	]

	recipe = {feat: os.path.join('/misc/kfdata01/kf_grp/hrwang/socialKG/CMU-MultimodalSDK/cmumosi', feat) + '.csd' for feat in features}
	label_recipe = {label_field: os.path.join('/misc/kfdata01/kf_grp/hrwang/socialKG/CMU-MultimodalSDK/cmumosi', label_field + '.csd')}

	dataset = md.mmdataset(recipe)
	dataset.add_computational_sequences(label_recipe, destination=None)
	return dataset


def get_mosi_label(dataset, video_id, segement_id):
	label_field = 'CMU_MOSI_Opinion_Labels'
	return dataset[label_field][video_id]['features'][segement_id][0]

if __name__ == '__main__':
	dataset = load_dataset()
	label = get_mosi_label(dataset, '0h-zjBukYpk', 2)
	label = get_mosi_label(dataset, '0h-zjBukYpk', 24)
	print(label)
