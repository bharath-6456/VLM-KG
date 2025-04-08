# VLM-KG

python3 -m venv clip_env
source clip_env/bin/activate
pip install torch torchvision torchaudio

pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git

to load embeddings
pip install networkx matplotlib scikit-learn

For yolo object detection
pip install ultralytics

Text Concept Extraction using spaCy + KeyBERT
pip install spacy keybert nltk
python3 -m nltk.downloader punkt
python3 -m spacy download en_core_web_sm
