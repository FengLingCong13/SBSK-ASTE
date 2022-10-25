## SBSK-ASTE

Code for Improving Span-based Aspect Sentiment Triplet Extraction with Abundant Syntax Knowledge

### Usage

- Install data and requirements: `bash setup.sh`
- Run training and evaluation on GPU 0: `bash aste/main.sh 0`
- Training config (10 epochs): [training_config/aste.jsonnet](training_config/aste.jsonnet)
- Modified data reader: [span_model/data/dataset_readers/span_model.py](span_model/data/dataset_readers/span_model.py)
- Modeling code: [span_model/models/span_model.py](span_model/models/span_model.py)

### Model Architecture
![image](https://user-images.githubusercontent.com/45933255/197764098-3c8becf6-33ef-42f8-929e-2a880c9fe9f8.png)

