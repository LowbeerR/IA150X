import os
import time
import winpty


def run_isg(mode, path):
    print("Starting ISG")
    docker_path = r"C:\Program Files\Docker\Docker\resources\bin\docker.exe"
    docker_command = [
        docker_path,
        "run",
        "-it",
        "--rm",
        "-v",
        f"C:/Users/{os.getlogin()}/Documents/GitHub/Infinite-Storage-Glitch:/home/Infinite-Storage-Glitch",
        "isg",
        "./target/release/isg_4real"
    ]
    process = winpty.PtyProcess.spawn(docker_command)
    time.sleep(1)
    process.write('\r')
    time.sleep(1)
    process.write(f'{path}\r')
    time.sleep(1)
    for i in range(mode):
        process.write('\033[B')
        time.sleep(1)
    process.write('\r')
    process.wait()
    print("ISG terminated")
