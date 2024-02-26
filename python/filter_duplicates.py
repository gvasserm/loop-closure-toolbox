import fiftyone as fo
import fiftyone.zoo as foz
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

dataset = fo.Dataset.from_dir("/home/gvasserm/dev/rtabmap/data/samples/", dataset_type=fo.types.UnlabeledImageDataset)
model = foz.load_zoo_model("mobilenet-v2-imagenet-torch")
embeddings = dataset.compute_embeddings(model)
print(embeddings.shape)

similarity_matrix = cosine_similarity(embeddings)

print(similarity_matrix.shape)
print(similarity_matrix)

n = len(similarity_matrix)

similarity_matrix = similarity_matrix - np.identity(n)

id_map = [s.id for s in dataset.select_fields(["id"])]

for idx, sample in enumerate(dataset):
    sample["max_similarity"] = similarity_matrix[idx].max()
    sample.save()

session = fo.launch_app(dataset)