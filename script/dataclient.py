import argparse
import os
import shutil
import subprocess
import tempfile
import time

import paramiko
from getpass import getpass

from scp import SCPClient

HOST = "mipt-client.atp-fivt.org"


def run_docker(command: str, echo: bool = True):
    if echo:
        print(f" docker > {command}")

    result = subprocess.run(command.split(" "))
    if result.returncode != 0:
        raise ValueError("Failed to execute docker command")


def run_ssh(
    command: str, ssh: paramiko.SSHClient, skippable: bool = False, echo: bool = True
):
    if echo:
        print(f" ssh > {command}")

    _, stdout, stderr = ssh.exec_command(command)
    err = stderr.read().decode("utf-8")
    if err:
        print(err)
        if not skippable:
            raise ValueError("Command failed")

    out = stdout.read().decode("utf-8")
    if out:
        print(out)


def upload_logs_to_hdfs(command_args):
    password = getpass(f"{HOST} password for {args.user}:")

    target_hdfs_dir = f"/user/{command_args.user}/{command_args.hdfs_dir[0]}"
    print(
        f"## Uploading data from {command_args.log_dir} to {target_hdfs_dir} on behalf of {command_args.user} for {command_args.recommender} recommenders"
    )

    local_tmp_dir = tempfile.mkdtemp()
    remote_temp_dir = "tmp/" + str(int(time.time()))

    recommenders = [
        f"botify-recommender-{i}" for i in range(1, command_args.recommender + 1)
    ]

    ssh = None
    try:
        for recommender in recommenders:
            path = os.path.join(local_tmp_dir, recommender)
            run_docker(
                f"docker cp {recommender}:{command_args.log_dir} {path}", args.echo,
            )

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(
            hostname=HOST, username=command_args.user, password=password, port=22
        )
        scp = SCPClient(ssh.get_transport())

        run_ssh("mkdir -p " + remote_temp_dir, ssh, echo=args.echo)

        paths = []
        for folder, subs, files in os.walk(local_tmp_dir):
            paths += [os.path.join(folder, sub) for sub in subs]

        scp.put(paths, remote_path=remote_temp_dir, recursive=True)

        run_ssh(f"hadoop fs -mkdir -p {target_hdfs_dir}", ssh, echo=args.echo)

        if command_args.cleanup:
            run_ssh(
                f"hadoop fs -rm -r {target_hdfs_dir}/*",
                ssh,
                skippable=True,
                echo=args.echo,
            )

        run_ssh(
            f"hadoop fs -put {remote_temp_dir}/* {target_hdfs_dir}", ssh, echo=args.echo
        )
    finally:
        shutil.rmtree(local_tmp_dir)

        if ssh is not None:
            run_ssh(f"rm -r {remote_temp_dir}", ssh, echo=args.echo)
            ssh.close()


def download_logs(command_args):
    local_dir = command_args.local_dir[0]

    print(
        f"## Downloading data from {command_args.log_dir} to {local_dir} for {command_args.recommender} recommenders"
    )

    if os.path.exists(local_dir):
        os.rmdir(local_dir)
    os.makedirs(local_dir)

    recommenders = [
        f"botify-recommender-{i}" for i in range(1, command_args.recommender + 1)
    ]

    for recommender in recommenders:
        path = os.path.join(local_dir, recommender)
        run_docker(
            f"docker cp {recommender}:{command_args.log_dir} {path}", args.echo,
        )

    print(f"## Finished: {os.listdir(local_dir)}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--user", help=f"Login used to access {HOST}", type=str, required=False,
    )
    parser.add_argument(
        "--recommender",
        help="Number of recommender service docker containers",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--echo",
        help="Print command before executing it",
        action="store_true",
        default=True,
    )

    subparsers = parser.add_subparsers()

    log_2_hdfs = subparsers.add_parser(
        "log2hdfs", help="Upload recommender logs to HDFS"
    )
    log_2_hdfs.add_argument(
        "--cleanup", help="clean hdfs dir before", action="store_true", default=False
    )
    log_2_hdfs.add_argument(
        "--log-dir",
        help="Directory containing the uploaded log files",
        type=str,
        default="/app/log/.",
    )
    log_2_hdfs.add_argument(
        "hdfs_dir",
        help="Target HDFS directory (rooted at user's home)",
        type=str,
        nargs=1,
    )
    log_2_hdfs.set_defaults(func=upload_logs_to_hdfs)

    log2_local = subparsers.add_parser(
        "log2local", help="Upload recommender logs to local drive"
    )
    log2_local.add_argument(
        "--log-dir",
        help="Directory containing the uploaded log files",
        type=str,
        default="/app/log/.",
    )
    log2_local.add_argument(
        "local_dir", help="Target local directory", type=str, nargs=1,
    )
    log2_local.set_defaults(func=download_logs)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.func(args)
