## Sentiment Analysis Application

Welcome to our Python desktop application for sentiment analysis, Using our model XLM-RoBERTa-German-Sentiment model. Our application provides sentiment analysis across 8 languages, with focus on the German language. This tool is for anyone interested in uncovering insights from textual data.

You can refer to the sentiment analysis model details on [Hugging Face](https://huggingface.co/ssary/XLM-RoBERTa-German-sentiment)

Refer to the [paper](https://drive.google.com/file/d/1xg7zbCPTS3lyKhQlA2S4b9UOYeIj5Pyt/view?usp=drive_link) for more information about the training methodology and the results of the model used in the application and the Design and Diagrams of the application.

## Features

- Sentiment analysis across 8 languages, specializing in German with 87% F1 score.
- Utilizes the robust XLM-RoBERTa architecture and fine tuned with German dataset contains many domains, the dataset is subset of [German Bert's Dataset](https://github.com/oliverguhr/german-sentiment).
- The Model-View-Controller (MVC) design pattern has been implemented to separate thefront-end and back-end code into different components, this separation facilitates more manageable changes and updates to each side, reducing the risk of interference between the two components.

- For the database, I'm using PostgreSQL and implementing Object-Relational Mapping (ORM) which adds a layer of abstraction over the database operations, allowing to work with the data and tables as objects rather than SQL queries, this is implemented using SQLAlchemy in Python, also it contains some useful abstract database operations creating a set of classes that encapsulate all database-related interactions, this abstraction allows for more manageable, modular, and maintainable code.

## Installation
1. Clone the repository:
```bash
git clone https://github.com/ssary/German-Sentiment-Analysis
```

2. Change the directory to the repo folder with:
```bash
cd '.\XLM-RoBERTa model\'
```

3. Create virtual environment "myenv" with:
```bash
python -m venv myenv.
```
4. To activate the virtual environment: 
```bash
source myenv/Scripts/activate
```
or with:
```bash 
source myenv/bin/activate
```

5. Install dependencies:
```bash 
pip install -r requirements.txt
```
6. Change the Database URL in the settings.env to your database URL, the format of postgresql is ```DATABASE_URL="postgresql://USERNAME:YOUR_PASSWORD@HOST:PORT/DATABASE_NAME```
where you change USERNAME, YOUR_PASSWORD, HOST usually is localhost, PORT is usually 5432 and DATABASE_NAME with the corresponding values.

8. Launch the application:
```bash
python controller.py
```

## Usage

Add review text with any of these 8 language (German, Arabic, English, French, Hindi, Italian, portuguese, Spanish) and you'll get positive, negative or neutral for the review.

![img](https://i.ibb.co/B2xbbSp/initial-design-figma.png)

## Training Scripts

- [Fine tuning the model on HPC](https://github.com/ssary/German-Sentiment-Analysis/blob/main/scripts/finetuning.py)
- [Testing our 200K model](https://github.com/ssary/German-Sentiment-Analysis/blob/main/scripts/test-200k-model-on-various-datasets.ipynb)
- [Testing Before Fine Tuning on German Bert Dataset](https://github.com/ssary/German-Sentiment-Analysis/blob/main/scripts/test-xlm-t-on-germanbert-data.ipynb)
- [Changing German Bert dataset structure to fit for the training](https://github.com/ssary/German-Sentiment-Analysis/blob/main/scripts/change-germeval-structure.ipynb)

## Results
Here is comparison between the model before fine tuning and after fine tuning, the F1 accuracy increased by 10%, achieving 87% accuracy on German Bert Dataset.

![F1 accuracy between the 4 models](https://i.ibb.co/DRRtxzN/f1-base-finetuned.png)
## Dataset Acknowledgment

We extend our heartfelt gratitude to Oliver Guhr for developing the German-language dataset utilized in training our model. This dataset, available on GitHub, has been instrumental in enhancing our model's performance. For more details on the dataset [this GitHub repository](https://github.com/oliverguhr/german-sentiment)

## References

- For more on the XLM-RoBERTa architecture and its advantages, see the [RoBERTa paper](https://arxiv.org/abs/1907.11692).
- Our model's fine-tuning and training are based on the principles outlined in the [xlm-t paper](https://arxiv.org/abs/2104.12250).

## Contact

For any inquiries or further information, feel free to contact me at sarynasser1@gmail.com.
