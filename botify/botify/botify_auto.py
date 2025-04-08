import subprocess
import sys
import time
import requests
from pathlib import Path
from datetime import datetime


def wait_for_service(url: str, timeout: int = 100, interval: int = 1) -> bool:
    """Ожидает доступности сервиса по URL"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                print(f"✅ Сервис {url} доступен")
                return True
        except (requests.exceptions.RequestException, requests.exceptions.ConnectionError):
            print(f"⏳ Ожидание сервиса {url}... (таймаут через {int(timeout - (time.time() - start_time))}s)")
            time.sleep(interval)
    print(f"❌ Сервис {url} не стал доступен за {timeout} секунд")
    return False


def run_command(cmd: str, error_msg: str, check_service: str = None) -> None:
    """Запускает команду с проверкой сервиса при необходимости"""
    full_cmd = f"""
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate lightfm_env
    {cmd}
    """

    print(f"🚀 Выполняется: {cmd}")
    process = subprocess.Popen(
        ["bash", "-c", full_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())

    if process.returncode != 0:
        print(f"❌ {error_msg} (код: {process.returncode})")
        sys.exit(1)

    if check_service:
        if not wait_for_service(check_service):
            sys.exit(1)


def prepare_directory(dir_path: str) -> None:
    """Подготавливает рабочую директорию"""
    path = Path(dir_path)

    if path.exists():
        if any(path.iterdir()):
            print(f"⚠️ Директория {dir_path} не пустая")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = f"{dir_path}_backup_{timestamp}"

            # Ищем уникальное имя для бэкапа
            counter = 1
            while Path(backup_dir).exists():
                backup_dir = f"{dir_path}_backup_{timestamp}_{counter}"
                counter += 1

            print(f"🔄 Переименовываю в {backup_dir}")
            try:
                path.rename(backup_dir)
            except Exception as e:
                print(f"❌ Ошибка переименования: {e}")
                sys.exit(1)
        else:
            print(f"📂 Директория {dir_path} существует и пуста")
            return

    # Создаем новую директорию
    print(f"📂 Создаю директорию: {dir_path}")
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"❌ Ошибка создания директории: {e}")
        sys.exit(1)


if __name__ == "__main__":
    num_recommenders = 2
    num_sims = 10000
    num_processes = 8
    data_dir = "hw_2/my_best_recommender_data"
    full_data_path = f"/mnt/c/Users/denis/PycharmProjects/recsys-course-spring-2025/rec_sys_data/{data_dir}"
    # full_data_path = f"/mnt/c/Users/denis/PycharmProjects/recsys-course-spring-2025/rec_sys_data/{data_dir}"

    prepare_directory(full_data_path)

    commands = [
        (
            f"cd /mnt/c/Users/denis/PycharmProjects/recsys-course-spring-2025/botify && "
            f"docker-compose up -d --build --force-recreate --scale recommender={num_recommenders}",
            "Ошибка Docker",
            "http://localhost:5001"
        ),
        (
            f"cd /mnt/c/Users/denis/PycharmProjects/recsys-course-spring-2025/sim && "
            f"python -m sim.run --episodes {num_sims} --config config/env.yml multi --processes {num_processes}",
            "Ошибка симуляции",
            None
        ),
        (
            f'cd /mnt/c/Users/denis/PycharmProjects/recsys-course-spring-2025/script && '
            f'python dataclient.py --recommender {num_recommenders} log2local '
            f'"{full_data_path}"',
            "Ошибка экспорта",
            None
        )
    ]

    for cmd, err_msg, check_url in commands:
        run_command(cmd, err_msg, check_url)

    print("✅ Все этапы выполнены!")
