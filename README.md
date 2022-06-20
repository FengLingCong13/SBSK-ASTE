## SBSK-ASTE

Code for Improving Span-based Aspect Sentiment Triplet Extraction with Abundant Syntax Knowledge

### Usage

- Install data and requirements: `bash setup.sh`
- Run training and evaluation on GPU 0: `bash aste/main.sh 0`
- Training config (10 epochs): [training_config/aste.jsonnet](training_config/aste.jsonnet)
- Modified data reader: [span_model/data/dataset_readers/span_model.py](span_model/data/dataset_readers/span_model.py)
- Modeling code: [span_model/models/span_model.py](span_model/models/span_model.py)

### Model Architecture
![1655730478077](https://user-images.githubusercontent.com/45933255/174608680-363499c2-557a-4e09-a0c3-ee5126b9a410.jpg)

