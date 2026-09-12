import fiftyone as fo
import fiftyone.zoo as foz

fo.config.dataset_zoo_directory = '../images/safe'
dataset = foz.load_zoo_dataset("coco-2017", splits=["validation"])