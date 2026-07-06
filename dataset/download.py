from roboflow import Roboflow

rf = Roboflow(api_key="ZmjUDQ47QznzeDXFn12G")

project = rf.workspace("academic-7a9qr").project("kitti-i65zf")

versions_list = project.versions()

if len(versions_list) > 0:
    first_version = versions_list[0]

    v_num = getattr(first_version, 'version', None) or first_version.id.split('/')[-1]

    print(f"Попытка скачивания версии: {v_num}")
    dataset = project.version(int(v_num)).download("yolov8")
else:
    print("В проекте не найдено доступных версий.")
