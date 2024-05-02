# Retrival Augmented Generation (RAG) With and Without LangChain

This project entails three parts:

1. Performing a simple QA using LangChain
2. Performing RAG using the LangChain pipeline and FAISS
3. Performing RAG without LangChain using embeddings created from a Sentence Transformers model that are stored in FAISS

## Provided files

- `questions.csv`: The train data for the project
- `passages.csv`: The test data for the project
- `val_questions.csv`: The notebook used to develop the model and code for the project
- `main.py`: The python file that contains the final code for the project. It contains an argparse mechanism that can be used to train and test the model. Sample usage of argparse for this code is defined below in the **Sample Run Commands** section.
- `requirements.txt`: Contains the list of require packages to set up an adequate environment for the project.
- `score.py`: Scoring script

## Setup

***I did add an os.system command for the first commands, but keeping this here just in case.***

To run this code, make sure to do the following first in cli:

```
pip install --quiet -U langchain-community

pip install -r requirements.txt
```

Then, make sure to set the global variable `hf_token` as a HuggingFace token. **This is necessary for the code to run**

## Sample Run Commands
```
# Run with no RAG
python main.py --questions ./data/val_questions.csv --output val_no_rag.csv

# Run with RAG (with langchain embeddings)
python main.py --questions ./data/val_questions.csv --rag --langchain --passages ./data/passages.csv --output val_rag_langchain.csv

# Run with RAG (with custom embeddings)
python main.py --questions ./data/val_questions.csv --rag --passages ./data/passages.csv --output val_rag.csv

# Checking Validation Scores
python score.py --golds /path/to/val_questions.csv --preds /path/to/preds.csv


```

