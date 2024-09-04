import json
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from graphs import plot_training_history, plot_test_metrics


def load_data(file_path):
    """
    Загрузка данных из JSON-файла и преобразование их в DataFrame.

    :param file_path: Путь к JSON-файлу.
    :return: DataFrame с колонками 'text', 'author', 'title'.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    poems = []
    for entry in data:
        poet = entry['poet_id']
        text = entry['content']
        title = entry.get('title', 'Unknown')
        poems.append({"text": text, "author": poet, "title": title})

    df = pd.DataFrame(poems)
    return df


def split_text_to_fragments(df, min_length=100):
    """
    Разделение текстов на фрагменты длиной не менее 100 символов.

    :param df: DataFrame с колонками 'text', 'author', 'title'.
    :param min_length: Минимальная длина фрагмента текста.
    :return: DataFrame с разделёнными текстовыми фрагментами.
    """
    fragments = []
    for index, row in df.iterrows():
        text = row['text']
        author = row['author']
        title = row['title']
        for i in range(0, len(text), min_length):
            fragment = text[i:i + min_length]
            if len(fragment) >= min_length:
                fragments.append({"text": fragment, "author": author, "title": title})

    return pd.DataFrame(fragments)


def prepare_datasets(df, test_size=0.2, val_size=0.1):
    """
    Разделение данных на обучающую, тестовую и валидационную выборки.

    :param df: DataFrame с колонками 'text' и 'author'.
    :param test_size: Доля данных для тестовой выборки.
    :param val_size: Доля данных для валидационной выборки.
    :return: Три DataFrame: train_df, val_df, test_df.
    """
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['author'], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=val_size, stratify=train_df['author'], random_state=42)
    return train_df, val_df, test_df


def tokenize_data(df, tokenizer, max_length=128):
    """
    Токенизация текстов с использованием предобученного токенизатора BERT.

    :param df: DataFrame с текстами и метками.
    :param tokenizer: Предобученный токенизатор BERT.
    :param max_length: Максимальная длина текстовых последовательностей.
    :return: Токенизированные входные данные, attention masks и метки.
    """
    input_ids = []
    attention_masks = []

    for text in df['text']:
        encoded_data = tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # Добавляем [CLS] и [SEP]
            max_length=max_length,  # Ограничиваем длину текста
            padding='max_length',  # Добавляем padding до max_length
            truncation=True,  # Обрезаем текст, если он длиннее max_length
            return_attention_mask=True,  # Возвращаем маску внимания (attention mask)
            return_tensors='pt'  # Возвращаем тензоры PyTorch
        )

        input_ids.append(encoded_data['input_ids'])
        attention_masks.append(encoded_data['attention_mask'])

    # Преобразуем списки в тензоры
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(df['author'].factorize()[0])

    return input_ids, attention_masks, labels


def create_dataloader(input_ids, attention_masks, labels, batch_size=16):
    """
    Создание DataLoader для подачи данных в модель.

    :param input_ids: Токенизированные входные данные.
    :param attention_masks: Маски внимания.
    :param labels: Метки классов.
    :param batch_size: Размер батча.
    :return: DataLoader для использования в процессе обучения модели.
    """
    data = TensorDataset(input_ids, attention_masks, labels)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return dataloader


def train_bert(model, train_dataloader, val_dataloader, epochs=4, learning_rate=2e-5, total_steps=None):
    """
    Обучение модели BERT.

    :param model: Модель BERT с полносвязным слоем.
    :param train_dataloader: DataLoader для обучающей выборки.
    :param val_dataloader: DataLoader для валидационной выборки.
    :param epochs: Количество эпох обучения.
    :param learning_rate: Начальная скорость обучения.
    :param total_steps: Общее количество шагов (если None, вычисляется автоматически).
    :return: Обученная модель, история обучения (потери и точность).
    """
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    if total_steps is None:
        total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    train_losses = []
    val_losses = []
    val_accuracies = []

    model.train()

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        total_train_loss = 0

        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            b_input_ids, b_input_mask, b_labels = batch
            model.zero_grad()

            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()

            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        print(f'Average training loss: {avg_train_loss}')

        model.eval()
        total_val_accuracy = 0
        total_val_loss = 0

        for batch in tqdm(val_dataloader, desc="Validation"):
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs.loss
                logits = outputs.logits

            total_val_loss += loss.item()

            preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
            labels = b_labels.cpu().numpy()
            total_val_accuracy += accuracy_score(labels, preds)

        avg_val_accuracy = total_val_accuracy / len(val_dataloader)
        avg_val_loss = total_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_accuracy)

        print(f'Validation loss: {avg_val_loss}, Validation accuracy: {avg_val_accuracy}')

        model.train()

    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies
    }

    return model, history


def freeze_bert_layers(model):
    """
    Заморозка весов BERT для обучения только полносвязного слоя.

    :param model: Модель BERT.
    """
    for param in model.bert.parameters():
        param.requires_grad = False


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional,
                            dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths=None):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)

        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        return self.fc(hidden)


def train_lstm(model, train_dataloader, val_dataloader, epochs=5, learning_rate=1e-3):
    """
    Обучение модели на основе LSTM.

    :param model: LSTM модель для обучения.
    :param train_dataloader: DataLoader для обучающей выборки.
    :param val_dataloader: DataLoader для валидационной выборки.
    :param epochs: Количество эпох обучения.
    :param learning_rate: Скорость обучения.
    :return: Обученная модель, история обучения (потери и точность).
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    val_accuracies = []

    model.train()

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        total_train_loss = 0

        for batch in tqdm(train_dataloader, desc="Training"):
            text, text_lengths, labels = batch
            model.zero_grad()

            predictions = model(text, text_lengths).squeeze(1)
            loss = criterion(predictions, labels)
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        print(f'Average training loss: {avg_train_loss}')

        model.eval()
        total_val_accuracy = 0
        total_val_loss = 0

        for batch in tqdm(val_dataloader, desc="Validation"):
            text, text_lengths, labels = batch

            with torch.no_grad():
                predictions = model(text, text_lengths).squeeze(1)
                loss = criterion(predictions, labels)

            total_val_loss += loss.item()
            preds = np.argmax(predictions.detach().cpu().numpy(), axis=1)
            labels = labels.cpu().numpy()
            total_val_accuracy += accuracy_score(labels, preds)

        avg_val_accuracy = total_val_accuracy / len(val_dataloader)
        avg_val_loss = total_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_accuracy)

        print(f'Validation loss: {avg_val_loss}, Validation accuracy: {avg_val_accuracy}')

        model.train()

    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies
    }

    return model, history


def evaluate_model(model, test_dataloader):
    """
    Оценка модели на тестовом наборе данных.

    :param model: Обученная модель.
    :param test_dataloader: DataLoader для тестового набора данных.
    :return: Средние значения функции потерь и точности.
    """
    model.eval()
    total_test_accuracy = 0
    total_test_loss = 0
    criterion = nn.CrossEntropyLoss()

    for batch in tqdm(test_dataloader, desc="Testing"):
        text, text_lengths, labels = batch

        with torch.no_grad():
            output = model(text, text_lengths)

            # Если модель BERT, получаем logits
            if isinstance(output, torch.nn.modules.container.Sequential) or isinstance(output, torch.Tensor):
                predictions = output
            else:
                predictions = output.logits

            loss = criterion(predictions, labels)

        total_test_loss += loss.item()
        preds = np.argmax(predictions.detach().cpu().numpy(), axis=1)
        labels = labels.cpu().numpy()
        total_test_accuracy += accuracy_score(labels, preds)

    avg_test_accuracy = total_test_accuracy / len(test_dataloader)
    avg_test_loss = total_test_loss / len(test_dataloader)

    print(f'Test loss: {avg_test_loss}, Test accuracy: {avg_test_accuracy}')
    return avg_test_loss, avg_test_accuracy


if __name__ == "__main__":
    # Загрузка данных
    file_path = 'data/classic_poems.json'
    df = load_data(file_path)

    # Разделение текстов на фрагменты
    df_fragments = split_text_to_fragments(df)

    # Подготовка обучающей, тестовой и валидационной выборок
    train_df, val_df, test_df = prepare_datasets(df_fragments)


    # Загрузка предобученного токенизатора BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Токенизация данных
    max_length = 64  # Максимальная длина последовательности
    train_inputs, train_masks, train_labels = tokenize_data(train_df, tokenizer, max_length)
    val_inputs, val_masks, val_labels = tokenize_data(val_df, tokenizer, max_length)
    test_inputs, test_masks, test_labels = tokenize_data(test_df, tokenizer, max_length)

    # Создание DataLoader для каждой выборки
    batch_size = 8
    train_dataloader = create_dataloader(train_inputs, train_masks, train_labels, batch_size)
    val_dataloader = create_dataloader(val_inputs, val_masks, val_labels, batch_size)
    test_dataloader = create_dataloader(test_inputs, test_masks, test_labels, batch_size)

    print(f"Обучающая выборка: {len(train_dataloader.dataset)} примеров")
    print(f"Валидационная выборка: {len(val_dataloader.dataset)} примеров")
    print(f"Тестовая выборка: {len(test_dataloader.dataset)} примеров")

    # Загрузка предобученной модели BERT с полносвязным слоем
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                          num_labels=len(train_df['author'].unique()))

    # Модель 1a: Заморозка весов BERT
    freeze_bert_layers(model)

    # Обучение модели (замороженные веса BERT)
    trained_model_1a = train_bert(model, train_dataloader, val_dataloader, epochs=4)

    # Fine-tuning (разморозка всех слоев и обучение всей модели)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                          num_labels=len(train_df['author'].unique()))
    # Обучение модели BERT с историей
    trained_model_1b, history_bert = train_bert(model, train_dataloader, val_dataloader, epochs=4)

    # Параметры модели
    vocab_size = tokenizer.vocab_size
    embedding_dim = 100
    hidden_dim = 256
    output_dim = len(train_df['author'].unique())
    n_layers = 2
    bidirectional = False
    dropout = 0.3

    # Создание модели стека LSTM
    model_lstm = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout)

    # Обучение модели стека LSTM с историей
    trained_lstm, history_lstm = train_lstm(model_lstm, train_dataloader, val_dataloader, epochs=5)

    # Создание модели двунаправленной LSTM
    bidirectional = True
    model_bilstm = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout)

    # Обучение модели двунаправленной LSTM с историей
    trained_bilstm, history_bilstm = train_lstm(model_bilstm, train_dataloader, val_dataloader, epochs=5)

    plot_training_history({
        'loss': history_bert['train_losses'],
        'val_loss': history_bert['val_losses']
    }, model_name="BERT")
    plot_training_history({
        'loss': history_lstm['train_losses'],
        'val_loss': history_lstm['val_losses']
    }, model_name="LSTM")
    plot_training_history({
        'loss': history_bilstm['train_losses'],
        'val_loss': history_bilstm['val_losses']
    }, model_name="BiLSTM")

    print("Evaluating BERT Model...")
    bert_test_loss, bert_test_accuracy = evaluate_model(trained_model_1b, test_dataloader)
    print("Evaluating LSTM Model...")
    lstm_test_loss, lstm_test_accuracy = evaluate_model(trained_lstm, test_dataloader)
    print("Evaluating BiLSTM Model...")
    bilstm_test_loss, bilstm_test_accuracy = evaluate_model(trained_bilstm, test_dataloader)

    plot_test_metrics({
        'Test Loss': bert_test_loss,
        'Test Accuracy': bert_test_accuracy
    }, model_name="BERT")
    plot_test_metrics({
        'Test Loss': lstm_test_loss,
        'Test Accuracy': lstm_test_accuracy
    }, model_name="LSTM")
    plot_test_metrics({
        'Test Loss': bilstm_test_loss,
        'Test Accuracy': bilstm_test_accuracy
    }, model_name="BiLSTM")
