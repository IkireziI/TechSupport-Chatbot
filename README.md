# TECH SUPPORT CHATBOT


# Tech Support Chatbot: Fine-tuned T5-small for Technical Assistance

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [How It Works](#how-it-works)
- [Technologies Used](#technologies-used)
- [Setup and Local Usage (Google Colab)](#setup-and-local-usage-google-colab)
- [Deployed Chatbot](#deployed-chatbot)
- [Project Structure](#project-structure)
- [Future Enhancements](#future-enhancements)
- [License](#license)
- [Author](#author)

## Project Overview

This project implements an AI-powered chatbot designed to provide automated assistance for common technical support inquiries. Leveraging advanced Natural Language Processing (NLP) techniques, the chatbot understands user questions and generates relevant, helpful responses, acting as a first line of support.

## Features

* **Intelligent Response Generation:** Generates human-like, coherent, and contextually relevant answers to tech support questions.
* **Specialized Knowledge:** Fine-tuned on a domain-specific dataset to address common technical issues.
* **Interactive Interface:** Provides a user-friendly web interface powered by Gradio for easy interaction.
* **Cloud Deployment:** Hosted permanently on Hugging Face Spaces for continuous accessibility.

## How It Works

The chatbot's core functionality is built upon a **Transformer-based sequence-to-sequence model** from the T5 (Text-to-Text Transfer Transformer) family. Specifically, the `t5-small` model was utilized due to its efficiency and strong performance.

1.  **Foundation Model (`t5-small`):** The project starts with a pre-trained `t5-small` model, which has learned extensive language patterns from a vast amount of text data. This model excels at converting any text-based problem (like question answering) into a text-to-text format.

2.  **Domain Specialization (Fine-tuning):** To adapt `t5-small` for tech support, it undergoes a crucial **fine-tuning** process. This involves training the model on a custom dataset containing pairs of `user_query` (technical support questions) and `chatbot_response` (corresponding detailed answers or troubleshooting steps). This specialization teaches the model the specific terminology, problem patterns, and appropriate solutions within the tech support domain.

3.  **Inference and Interaction:** When a user inputs a question through the Gradio interface, the fine-tuned model processes this input and generates a relevant output sequence (the answer).

## Technologies Used

* **Python:** The primary programming language.
* **Google Colab:** Cloud-based Jupyter notebook environment used for development and training.
* **Hugging Face Transformers:** Library providing the `t5-small` model, `AutoTokenizer`, `Trainer`, and `TrainingArguments`.
* **Hugging Face Datasets:** Library for efficient data loading and processing.
* **PyTorch:** The deep learning framework used for model training and inference.
* **Gradio:** A Python library for quickly building interactive web interfaces for machine learning models.
* **Hugging Face Spaces:** A platform for hosting machine learning demos permanently.
* **Pandas:** For data manipulation and CSV handling.
* **NumPy:** For numerical operations.
* **Evaluate & Rouge-score:** For evaluating model performance.

## Setup and Local Usage (Google Colab)

To run and experiment with this chatbot in your Google Colab environment:

1.  **Open the Notebook:** Open the `TechSupport-Chatbot_Model.ipynb` file in Google Colab.
2.  **Connect to Runtime:** Ensure you are connected to a Colab runtime (preferably with a GPU, though `t5-small` can run on CPU).
3.  **Run Cells Sequentially:** Execute all code cells in the notebook from top to bottom.
    * **Cell 1:** Installs necessary libraries (PyTorch, Transformers, Gradio, etc.). **(Remember to `Runtime -> Restart runtime` after this cell completes)**
    * **Cell 2:** Loads your tech support dataset.
    * **Cell 3:** Tokenizes the data and prepares it for model training.
    * **Cell 4-6:** Loads the `t5-small` model, defines training arguments, and starts the fine-tuning process.
    * **Cell 7-8:** Evaluates the trained model and saves it.
    * **Cell 9:** Launches a local Gradio interface for interaction.

## Deployed Chatbot

For immediate interaction without needing to run the Colab notebook, the chatbot is permanently deployed on Hugging Face Spaces:

**🔗 [Link to Your Hugging Face Space](https://huggingface.co/spaces/ikirezii/inesii)**



## Project Structure

This repository contains the following key files:

* `TechSupport-Chatbot_Model.ipynb`: The Google Colab notebook containing all the code for data loading, preprocessing, model fine-tuning, evaluation, and local Gradio deployment.
*  tech_support_dataset.csv
 : The dataset used for fine-tuning the T5 model, containing `user_query` and `chatbot_response` pairs.
* `app.py`: The Python script for deploying the Gradio chatbot on Hugging Face Spaces, which loads the model directly from the Hugging Face Hub.
* `requirements.txt`: Lists all Python library dependencies required to run `app.py` on Hugging Face Spaces.


## Future Enhancements

* **Expanded Dataset:** Continuously grow the training dataset with more diverse and specific tech support scenarios to improve model accuracy and coverage.
* **Larger Models:** Experiment with `t5-base` or `t5-large` for potentially higher quality responses, given sufficient computational resources.
* **Conversational Memory:** Implement techniques to allow the chatbot to maintain context across multiple turns in a conversation.
* **User Feedback Loop:** Integrate mechanisms for users to rate responses, providing valuable data for iterative model improvement.
* **Multi-modal Input:** Explore adding support for images or other media in queries.

## License

This project is licensed under the **[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)** 

## Author

**IkireziI**
**Inès IKIREZI**

https://github.com/IkireziI/TechSupport-Chatbot.git

