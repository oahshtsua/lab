"""Bootstraping development environment setup with Airflow."""

import airflow
from airflow import DAG
from airflow.operators.bash import BashOperator

dag = DAG(
    dag_id="dev_env_setup_dag",
    start_date=airflow.utils.dates.days_ago(0),
    schedule_interval=None,
)

install_git = BashOperator(
    task_id="install_git",
    bash_command="echo $PASSWORD | sudo -S dnf install git",
    dag=dag,
)

clone_dotfiles_repo = BashOperator(
    task_id="clone_dotfiles_repo",
    bash_command="git clone https://github.com/ashshuota/infra.git $HOME/infra",
    dag=dag,
)

packages = [
    "neovim",
    "starship",
    "stow",
    "tmux",
    "trash-cli",
]

install_packages = BashOperator(
    task_id="install_packages",
    bash_command=f"echo $PASSWORD | sudo -S dnf install {' '.join(packages)}",
    dag=dag,
)

stow_dirs = [
    "bin",
    "nvim",
    "starship",
    "tmux",
]
stow_dotfiles = BashOperator(
    task_id="stow_dotfiles",
    bash_command=f"cd $HOME/infra/dotfiles/ && stow -t ~ {' '.join(stow_dirs)}",
    dag=dag,
)

tasks = [install_git, [clone_dotfiles_repo, install_packages], stow_dotfiles]
airflow.models.baseoperator.chain(*tasks)
