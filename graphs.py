import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid", font="DejaVu Sans", rc={"axes.labelsize": 14, "axes.titlesize": 16})


def plot_training_history(history, model_name):
    """
    Построение графика изменения потерь (loss) на обучении и валидации по эпохам.

    Аргументы:
        history (dict): Словарь, содержащий значения потерь и метрик по эпохам.
        model_name (str): Название модели, которое будет использовано в заголовке графика и имени файла.

    Возвращает:
        None: График сохраняется в файл и не возвращает никаких значений.
    """

    plt.figure(figsize=(12, 8))
    palette = sns.color_palette("Set2")
    plt.plot(history['loss'], label='Потери на обучении', color=palette[0], linewidth=2)
    plt.plot(history['val_loss'], label='Потери на валидации', color=palette[1], linewidth=2)

    plt.title(f'{model_name} - Потери', fontsize=16, fontweight='bold')
    plt.xlabel('Эпохи', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    model_name_sanitized = model_name.replace(" ", "_")
    file_path = f'img/graphs_img/{model_name_sanitized}_training_history.png'
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'График сохранён в {file_path}')


def plot_test_metrics(metrics, model_name):
    """
    Построение горизонтального столбчатого графика для отображения метрик на тестовых данных.

    Аргументы:
        metrics (dict): Словарь, где ключи — названия метрик, а значения — их значения.
        model_name (str): Название модели, которое будет использовано в заголовке графика и имени файла.

    Возвращает:
        None: График сохраняется в файл и не возвращает никаких значений.
    """

    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    plt.figure(figsize=(12, 8))
    palette = sns.color_palette("Spectral", len(metric_values))
    bars = plt.barh(metric_names, metric_values, color=palette)

    plt.title(f'{model_name} - Метрики на тестовых данных', fontsize=16, fontweight='bold')
    plt.xlabel('Значение', fontsize=14)
    for bar in bars:
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{bar.get_width():.4f}', va='center', fontsize=14, fontweight='bold')
    plt.tight_layout()

    model_name_sanitized = model_name.replace(" ", "_")
    file_path = f'img/graphs_img/{model_name_sanitized}_test_metrics.png'
    plt.savefig(file_path, dpi=300)
    plt.close()
    print(f'График сохранён в {file_path}')
