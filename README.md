# -LINGUA-AI-Text-to-Text-Translation-Using-NLP
A text to text translation model 
**Break Language Barriers with Intelligent Translation**

LINGUA AI is an NLP-based text translation application that converts text from one language into another using Sequence-to-Sequence (Seq2Seq) Transformer models. It uses pretrained MarianMT models from Hugging Face and supports fine-tuning on parallel language datasets.

The application provides a simple and interactive interface where users can select languages, enter text, and receive translated output instantly.

---

## 🚀 Project Overview

The primary objective of this project is to develop an intelligent language translation system using Natural Language Processing and Deep Learning.

The system processes user-provided text, tokenizes the input, passes it through a Transformer-based translation model, and generates translated text in the selected target language.

### Supported Languages

- 🇬🇧 English → 🇮🇳 Hindi
- 🇮🇳 Hindi → 🇬🇧 English

The architecture is designed to support additional language pairs in future versions.

---

## ✨ Features

- **Text Translation:** Translate text between supported languages.
- **Language Selection:** Select source and target languages.
- **Transformer Architecture:** Uses Sequence-to-Sequence MarianMT models.
- **NLP Preprocessing:** Cleans and prepares parallel text datasets.
- **Model Fine-Tuning:** Supports training pretrained models on custom datasets.
- **BLEU Evaluation:** Measures translation quality using SacreBLEU.
- **Interactive GUI:** Clean and responsive Streamlit interface.
- **Translation History:** Maintains translation history during the current session.
- **Copy Translation:** Easily copy translated text.
- **Error Handling:** Handles empty input and unsupported language pairs.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | Deep learning framework |
| Hugging Face Transformers | Pretrained translation models |
| MarianMT | Sequence-to-Sequence translation |
| OPUS | Parallel translation dataset |
| Pandas | Dataset preprocessing |
| NumPy | Numerical operations |
| Streamlit | User interface |
| SacreBLEU | Translation evaluation |
| Matplotlib | Training visualization |
| SentencePiece | Text tokenization |

---

## 🧠 Model Architecture

This project uses a Sequence-to-Sequence Transformer architecture.

The model consists of:

1. **Encoder:** Processes the input sentence and creates contextual representations.
2. **Decoder:** Generates the translated sentence token by token.
3. **Attention Mechanism:** Helps the model focus on relevant input tokens during translation.

### Translation Workflow

```text
       User Input
            |
            v
     Language Selection
            |
            v
     Text Preprocessing
            |
            v
        Tokenization
            |
            v
    Transformer Encoder
            |
            v
    Transformer Decoder
            |
            v
     Translated Output
            |
            v
       Streamlit GUI
```

### Pretrained Models

| Translation Direction | Model |
|---|---|
| English → Hindi | Helsinki-NLP/opus-mt-en-hi |
| Hindi → English | Helsinki-NLP/opus-mt-hi-en |

Model source: [Hugging Face – Helsinki NLP](https://huggingface.co/Helsinki-NLP)

---

## 📂 Project Structure

```text
text_translation_project/
│
├── app.py
├── config.py
├── requirements.txt
├── README.md
│
├── dataset/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── en_hi/
│   └── hi_en/
│
├── src/
│   ├── dataset_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── translation.py
│   ├── evaluate.py
│   └── model_manager.py
│
├── notebooks/
│   └── model_training.ipynb
│
├── outputs/
│   ├── evaluation/
│   └── training_logs/
│
└── assets/
    └── custom.css
```

---

## 📊 Dataset

The project uses parallel text datasets for machine translation.

### OPUS Dataset

OPUS provides multilingual parallel corpora containing translated sentences in different languages.

Dataset source: https://opus.nlpl.eu/

### Tatoeba Dataset

Tatoeba provides multilingual example sentences and their translations.

Dataset source: https://tatoeba.org/

### Dataset Preprocessing

The preprocessing pipeline includes:

- Removing missing values.
- Removing duplicate sentence pairs.
- Cleaning unnecessary whitespace.
- Unicode normalization.
- Filtering invalid or excessively long sentences.
- Tokenization.
- Training, validation and testing split.

Dataset distribution:

| Dataset | Percentage |
|---|---:|
| Training | 80% |
| Validation | 10% |
| Testing | 10% |

---

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/text-translation-project.git
```

### 2. Navigate to the Project Directory

```bash
cd text-translation-project
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

If you have not created requirements.txt yet, install the main dependencies:

```bash
pip install torch transformers sentencepiece streamlit pandas numpy sacrebleu matplotlib scikit-learn datasets accelerate
```

---

## ▶️ Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

Default local address:

```text
http://localhost:8501
```

Select the source and target languages, enter your text, and click the Translate button.

**Note:** The first execution may take additional time because pretrained models need to be downloaded from Hugging Face.

---

## 🏋️ Model Fine-Tuning

The project supports fine-tuning pretrained MarianMT models using parallel translation datasets.

General training workflow:

1. Download the OPUS dataset.
2. Preprocess the dataset.
3. Load the pretrained MarianMT model.
4. Tokenize source and target sentences.
5. Prepare training and validation datasets.
6. Configure the optimizer and training parameters.
7. Fine-tune the model.
8. Save the trained model checkpoints.

Run the training script:

```bash
python -m src.train
```

For GPU-based training, Google Colab can be used.

Google Colab: https://colab.research.google.com/

---

## 📈 Model Evaluation

Translation quality is evaluated using the BLEU (Bilingual Evaluation Understudy) score.

BLEU compares machine-generated translations against reference translations.

Install SacreBLEU:

```bash
pip install sacrebleu
```

Run the evaluation script:

```bash
python -m src.evaluate
```

The evaluation module is designed to report:

- BLEU score.
- Training loss.
- Validation loss.
- Testing results.
- Translation quality comparisons.

Actual evaluation results should be added here after running the evaluation on the held-out test dataset.

---

## 🖥️ Application Interface

The application includes:

- Source language selection.
- Target language selection.
- Text input area.
- Translate button.
- Translated output display.
- Copy translation option.
- Clear text functionality.
- Translation history.

---

## 🔮 Future Enhancements

- Add support for more Indian languages.
- Support English to Tamil, Telugu, Bengali and Marathi translation.
- Integrate multilingual Transformer models.
- Add speech-to-text translation.
- Add text-to-speech functionality.
- Develop a mobile-friendly interface.
- Add translation history storage.
- Deploy the application online.
- Improve translation quality using larger parallel datasets.

---

## 🎯 Learning Outcomes

This project demonstrates practical implementation of:

- Natural Language Processing (NLP).
- Machine Translation.
- Sequence-to-Sequence Learning.
- Transformer Architecture.
- Encoder-Decoder Networks.
- Attention Mechanisms.
- Tokenization and Text Preprocessing.
- Transfer Learning and Fine-Tuning.
- Model Evaluation using BLEU.
- Deep Learning Application Development.

---

## 🤝 Contributing

Contributions are welcome!

If you would like to improve the project:

1. Fork the repository.
2. Create a new feature branch.
3. Commit your changes.
4. Push the branch.
5. Submit a Pull Request.

---

## 📜 License

This project is developed for educational and academic purposes.

You can add an MIT License if you want to distribute the project under that license.

---

## 👨‍💻 Author

**Your Name**

GitHub: https://github.com/YOUR_USERNAME

---

⭐ If you find this project useful, consider giving the repository a star!

**LINGUA AI – Breaking Language Barriers Through Artificial Intelligence.**
