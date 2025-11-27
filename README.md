<h1 align="center">SEMANTIC BOOK RECOMMENDER </h1>

![Home Page](./assets/image.png)
![Book Recommended](./assets/image-1.png)

<br/>

### Libraries, Tools and Technologies

![My Skills](https://skillicons.dev/icons?i=pycharm,vscode,git,github,python)

|                                                                                                       |                                                                                                                   |                                                                                                                      |                                                                                                                |
| ----------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| ![LLM](https://img.shields.io/badge/LLM-Powered-blue?style=for-the-badge&logo=openai&logoColor=white) | ![HuggingFace](https://img.shields.io/badge/🤗_Hugging_Face-92021E?style=for-the-badge&logoColor=000)             | ![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)                | ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)       |
| ![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white) | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)             | ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)                   | ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)  |
| ![Seaborn](https://img.shields.io/badge/Seaborn-9cf?style=for-the-badge&logo=python&logoColor=white)  | ![LangChain](https://img.shields.io/badge/🦜_LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white) | ![Transformers](https://img.shields.io/badge/Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000) | ![ipywidgets](https://img.shields.io/badge/ipywidgets-F37626?style=for-the-badge&logo=jupyter&logoColor=white) |

<br/>

### Index

| Content               | location                                  |
| --------------------- | ----------------------------------------- |
| Introduction          | [intro](#introduction)                    |
| Features              | [features](#features)                     |
| Project Contents      | [folder-contents](#project-contents)      |
| File Description      | [file-description](#files-description)    |
| Project Setup         | [setup](#project-setup)                   |
| Environment Variables | [set-env-file](#set-enviromnet-variables) |
| File Execution        | [file-execution](#file-execution)         |
| How it works?         | [working]()                               |

<br/>

### Introduction

This project provides a semantic book recommendation system that leverages advanced natural language processing techniques to suggest books based on user input. The system is built using:

-   [x] _**Python**_: The core programming language.
-   [x] _**HuggingFace**_: For embeddings and language models.
-   [x] _**LangChain**_: To manage language model interactions.
-   [x] _**Gradio**_: For creating an interactive web-based user interface.

<br/>

### Features

-   Semantic Search: Finds books that semantically match user queries.
-   Category Filtering: Allows filtering recommendations by book categories.
-   Emotional Tone Filtering: Filters recommendations based on the desired emotional tone.

<br/>

### Project Contents

| Project Content       | Files                                              |
| --------------------- | -------------------------------------------------- |
| Text Data Cleaning    | [data-cleaning](./data-exploration.ipynb)          |
| Semantic Analysis     | [semantic-analysis](./sentiment-analysis.ipynb)    |
| Text Classification   | [text-classification](./text-classification.ipynb) |
| Sentiment Analysis    | [sentiment-analysis](./sentiment-analysis.ipynb)   |
| Web App Dashboard     | [dashboard](./gradio-dashboard.py)                 |
| Environment Variables | [env-files](#set-environment-variables)            |
| Project Requirements  | [requirements](./requirements.txt)                 |

<br/>

### Files Description:

There are five important parts of this project:

**1. Text Data Cleaning**

-   Jupyter Notebook file: `data-exploration.ipynb`
-   Cleans and explores raw book data downloaded from Kaggle.

**2. Semantic (Vector) Search & Vector Database**

-   Jupyter Notebook file: `vector-search.ipynb`
-   Creates embeddings for each book description and stores them in a Chroma vector database for semantic search.
    Example query: “A book about a person seeking revenge.”

**3. Text Classification (Zero-Shot LLM)**

-   Jupyter Notebook file: `text-classification.ipynb`
-   Uses zero-shot classification to label each book as Fiction or Non-Fiction.

**4. Sentiment & Emotion Analysis**

-   Jupyter Notebook file: `sentiment-analysis.ipynb`
-   Extracts emotional tones such as Joy, Sadness, Fear, Anger, Surprise so users can sort books by tone.

**5. Interactive Gradio Dashboard**

-   _Python Dashboard File_: `gradio-dashboard.py`
-   A web application that lets users search for books by meaning, category, and emotion.

<br/>

### Project Setup

This project was developed using Python 3.12

1. Clone the repository

    ```
    git clone git@github.com:nihalsheikh/book_recommender.git
    cd book_recommender
    ```

<br/>

2. To run the project successfully, please ensure you install all the dependecis inthe requirements.txt file. A requirements.txt file, containing all necessary project dependencies, is provided in this repository for easy installation.
   You can install them using pip:

    ```
    pip install -r requirements.txt
    ```

<br/>

### Set Environment Variables

-   In the root dir of the project folder create a `.env` file
-   Add the necessary environment variables in the file:

    ```
    # For using OpenAI Model
    OPENAI_API_KEY=""

    # For using Google's AI Model
    GOOGLE_API_KEY=""

    # For using HuggingFace Model
    HUGGINGFACEHUB_API_TOKEN=""
    ```

<br/>

### File Execution

run/execute the files in the following order:

1. data-exploration.ipynb
2. vector-search.ipynb
3. text-classification.ipynb
4. sentiment-analysis.ipynb
5. gradio-dashboard.py

<br/>

### How it works?

-   Each book description is embedded using HuggingFace's `model_name="sentence-transformers/all-MiniLM-L6-v2`.

-   Vectors are stored in a local Chroma database.
-   A user’s query is embedded the same way and compared using cosine similarity.
-   The system filters by Fiction/Non-Fiction and sorts by emotion scores.
-   Results appear instantly in a Gradio gallery interface.
