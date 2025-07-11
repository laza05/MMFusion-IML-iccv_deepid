import pickle

import allegroai
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm

from inference import MMFusionDetector


def get_frames(dataset_name: str, version_name: str) -> list:

    temp_view = allegroai.DataView(auto_connect_with_task=False)
    temp_view.add_query(dataset_name=dataset_name, version_name=version_name)

    frames = temp_view.to_list()
    return frames


frames_test = get_frames("ml-id-iccv25-alteration", "2025-06-09-ae-original-dataset-test")
frames_train = get_frames("ml-id-iccv25-alteration", "2025-06-09-ae-original-dataset-train")
frames = frames_test + frames_train

inference = MMFusionDetector(
    config_path="experiments/ec_example_phase2.yaml",
    checkpoint_path="ckpt/model_zoo/early_fusion_detection.pth"
)


y_true = []
y_score = []
for frame in tqdm(frames):
    image_path = frame["front_cropped"].get_local_source()
    altered = frame.metadata.get("altered", [])
    label = 1 if len(altered) > 0 else 0

    score = inference.infer(image_path)
    y_true.append(label)
    y_score.append(score)

    print(f"Score: {score}, Label: {label}")

save_path = "results_early.pickle"
with open(save_path, "wb") as f:

    pickle.dump((y_true, y_score), f)
with open(save_path, "rb") as f:
    y_true, y_score = pickle.load(f)


y_binary = [1 if s > 0.5 else 0 for s in y_score]

f1 = f1_score(y_true, y_binary)
rocauc = roc_auc_score(y_true, y_score)
print(f"F1 Score: {f1}")
print(f"ROC AUC Score: {rocauc}")