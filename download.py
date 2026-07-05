from roboflow import Roboflow

# Инициализируем платформу с твоим ключом
rf = Roboflow(api_key="ZmjUDQ47QznzeDXFn12G")

# Подключаемся к проекту
project = rf.workspace("academic-7a9qr").project("kitti-i65zf")

# Получаем список версий
versions_list = project.versions()

if len(versions_list) > 0:
    # Берем первую доступную версию из списка
    first_version = versions_list[0]

    # Пытаемся взять номер через стандартное свойство .version
    # Если оно не сработает, код подстрахуется и вытащит ID версии напрямую
    v_num = getattr(first_version, 'version', None) or first_version.id.split('/')[-1]

    print(f"Попытка скачивания версии: {v_num}")
    dataset = project.version(int(v_num)).download("yolov8")
else:
    print("В проекте не найдено доступных версий.")