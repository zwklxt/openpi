"""
Script to convert Piper hdf5 data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/piper_real/convert_piper_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>

Examples usage:
uv run examples/piper_real/convert_piper_data_to_lerobot.py
    --raw_dir path/to/raw/data  #这里的raw_data，指的是piper采集的包含 *.hdf5数据的文件路径（数据处理的入口）
    --local_dir path/to/local/trans/result/data #这里指的是，转换后的数据，存放的本地位置，默认是None
    --repo_id <org>/<dataset> #这里指的是在hugface上的地址，如果存放到本地，可以不指定。


A6000:
#假设训练一个拿瓶子任务的数据集，可以如下组织和转换数据
uv run examples/piper_real/convert_piper_data_to_lerobot.py \
    --raw_dir /home/ricky/workspace/wenkai.zhang/pi0-piper/data/piper_raw/pick-up-the-bottle \
    --local_dir /home/ricky/workspace/wenkai.zhang/pi0-piper/data/piper_lerobot/pick-up-the-bottle \
    --repo_id amigos-robot/pick-up-the-bottle

H100:
uv run examples/piper_real/convert_piper_data_to_lerobot.py \
    --raw_dir /home/anker/wenkai.zhang/repo/pi0/data/piper_raw/grab_up_move_down \
    --local_dir /home/anker/wenkai.zhang/repo/pi0/data/piper_lerobot/grab_up_move_down \
    --repo_id amigos-robot/grab_up_move_down
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import os
import sys
# sys.path.insert(0, '/HostData/wenkai.zhang/repo/openpi/third_party/lerobot')

import h5py
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw
import numpy as np
import torch
import tqdm
import tyro


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()
6

def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    local_dir: str | Path | None = None,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = [
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
    ]
    cameras = [
        "cam_high",
        # "cam_low",
        "cam_left_wrist",
        "cam_right_wrist",
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    if Path(LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    if Path(local_dir).exists():
        shutil.rmtree(local_dir)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        root=local_dir, # noqa: SIM118
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def get_cameras(hdf5_files: list[Path]) -> list[str]:
    with h5py.File(hdf5_files[0], "r") as ep:
        # ignore depth channel, not currently handled
        return [key for key in ep["/observations/images"].keys() if "depth" not in key]  # noqa: SIM118


def has_velocity(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/qvel" in ep


def has_effort(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/effort" in ep


def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    camera_invalid = False
    for camera in cameras:
        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4

        if uncompressed:
            # load all images in RAM
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:
            import cv2

            # load one compressed image after the other in RAM and uncompress
            imgs_array = []
            for data in ep[f"/observations/images/{camera}"]:
                data = np.frombuffer(data, np.uint8)
                #这里判断下，data的值是否为空
                if data.size == 0:
                    camera_invalid = True
                    print(f"Warning:eposide:{ep}, Empty data for camera {camera}. Skipping this frame.")
                    imgs_array.append(np.zeros((480, 640, 3), dtype=np.uint8))
                else:
                    imgs_array.append(cv2.imdecode(data, 1))
            imgs_array = np.array(imgs_array)
            
        imgs_per_cam[camera] = imgs_array
        
    return imgs_per_cam,camera_invalid


def load_raw_episode_data(
    ep_path: Path,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    with h5py.File(ep_path, "r") as ep:
        invalid = False
        
        max_value= np.max(ep["/observations/qpos"][:])
        min_value = np.min(ep["/observations/qpos"][:])
        print(f"max_value: {max_value}, ep_path: {str(ep_path)}")
        
        if max_value > 3.14 or min_value < -3.14:
            invalid = True

        state = torch.from_numpy(ep["/observations/qpos"][:])
        # action = torch.from_numpy(ep["/action"][:])
        action = torch.from_numpy(ep["/observations/qpos"][:])


        velocity = None
        if "/observations/qvel" in ep:
            velocity = torch.from_numpy(ep["/observations/qvel"][:])

        effort = None
        if "/observations/effort" in ep:
            effort = torch.from_numpy(ep["/observations/effort"][:])

        imgs_per_cam,camera_invalid = load_raw_images_per_camera(
            ep,
            [
                "cam_high",
                # "cam_low",
                "cam_left_wrist",
                "cam_right_wrist",
            ],
        )
    
    invalid = True if invalid or camera_invalid else False

    return imgs_per_cam, state, action, velocity, effort, invalid 


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(hdf5_files))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]
        
        imgs_per_cam, state, action, velocity, effort, invalid= load_raw_episode_data(ep_path)
        
        if invalid:
            print(f"Warning:eposide:{ep_path}, Invalid data. Skipping this episode.")
            continue
        
        num_frames = state.shape[0]

        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
            }

            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]
                
            if velocity is not None:
                frame["observation.velocity"] = velocity[i]
            if effort is not None:
                frame["observation.effort"] = effort[i]

            dataset.add_frame(frame)

        dataset.save_episode(task=task)

    return dataset


def port_aloha(
    raw_dir: str | Path,
    repo_id: str | None = None,
    local_dir: str | Path | None = None,
    raw_repo_id: str | None = None,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    if type(raw_dir) is str:
        raw_dir = Path(raw_dir)

    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    print("raw_dir", raw_dir)
    if not raw_dir.exists():
        if raw_repo_id is None:
            raise ValueError("raw_repo_id must be provided if raw_dir does not exist")
        download_raw(raw_dir, repo_id=raw_repo_id)


    
    # 遍历raw_dir下的所有文件夹及子文件夹，找出所有的最低以及子文件夹的名称及路径，组成一个字典，key是文件夹名称，value是文件夹路径
    task_name_list = {}
    task_file_list = {}
    for root, dirs, files in os.walk(raw_dir):
        if not dirs:
            task_name = os.path.basename(root)
            task_name_list[task_name] = root
            hdf5_files = sorted(Path(root).glob("episode*.hdf5"))
            task_file_list[task_name] = hdf5_files
    
    print(f"Found {len(task_name_list)} tasks")

    print(list(task_name_list.keys()))
    
    hdf5_files_sample = task_file_list[list(task_name_list.keys())[0]]
    dataset = create_empty_dataset(
        repo_id,
        local_dir=local_dir,
        robot_type="mobile_aloha" if is_mobile else "aloha",
        mode=mode,
        has_effort=has_effort(hdf5_files_sample),
        has_velocity=has_velocity(hdf5_files_sample),
        dataset_config=dataset_config,
    )
        
    
    
    for i in range(len(task_name_list)):
        task_name = list(task_name_list.keys())[i]
        hdf5_files = task_file_list[task_name]
        dataset = populate_dataset(
            dataset,
            hdf5_files,
            task=task_name,
            episodes=episodes,
        )
    
    dataset.consolidate()

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(port_aloha)
