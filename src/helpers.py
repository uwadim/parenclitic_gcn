from pathlib import Path
from typing import Union


def get_create_path(path_str: Union[str, Path]) -> Path:
    """По заданному пути проверяет что есть такая директория, если нет,
    то создает ее, вместе c родительскими директориями

    :param path_str: Строка с путем

    :return: Объект Path
    """

    path = Path(path_str)
    # Проверяем, что директории есть, если нет, то создаем
    path.mkdir(parents=True, exist_ok=True)
    return path
