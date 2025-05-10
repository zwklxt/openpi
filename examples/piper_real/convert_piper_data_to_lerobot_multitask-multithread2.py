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
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Literal, List, Tuple

# 第三方库导入
import h5py
import numpy as np
import torch
import tqdm
import tyro
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME, LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


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
        action = torch.from_numpy(ep["/observations/qpos"][:])
        # action = torch.from_numpy(ep["/observations/qpos"][:])
        
        
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


def process_single_episode(ep_path, task, output_dir):
    imgs_per_cam, state, action, velocity, effort, invalid = load_raw_episode_data(ep_path)
    if invalid:
        print(f"Warning: episode {ep_path}, Invalid data. Skipping this episode.")
        return None
    
    dataset = create_empty_dataset(
        repo_id = "amigos-robot/tmp",
        local_dir=output_dir,
        robot_type="aloha",
        mode="image",
        has_effort=True,
        has_velocity=True,
        dataset_config=DEFAULT_DATASET_CONFIG,
    )
    
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
    
    output_path = Path(output_dir) / f"{ep_path.stem}.dataset"
    dataset.save_episode(task=task, save_path=output_path)
    return str(output_path)


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(hdf5_files))

    with ProcessPoolExecutor(max_workers=1) as executor:  # 单线程处理单个task内的所有episodes
        tmp_dir = Path(dataset.root) / task+"_tmp"
        futures_to_ep_idx = {executor.submit(process_single_episode, hdf5_files[ep_idx], task,tmp_dir): ep_idx 
                             for ep_idx in episodes}
        
        for future in tqdm(as_completed(futures_to_ep_idx), total=len(futures_to_ep_idx)):
            try:
                result = future.result()
                if result is None:
                    continue
                
                # 加载处理过的episode数据并合并到主dataset中
                dataset.load_episode(Path(result))
            except Exception as e:
                print(f"Episode processing failed with error: {e}")
    
    return dataset

def batch_process_tasks(tasks: List[Tuple[str, Path]], max_workers: int,task_file_list: dict) -> List[List[Tuple[str, Path]]]:
    """
    根据任务中的episode数量动态分配任务到不同的批次中，以确保每个批次的工作量尽可能平衡。
    """
    tasks_with_weights = [(task_name, task_path, len(task_file_list[task_name])) for task_name, task_path in tasks]
    tasks_with_weights.sort(key=lambda x: x[2], reverse=True)  # 按episode数量降序排序

    batches = [[] for _ in range(max_workers)]
    batch_weights = [0] * max_workers

    for task_name, task_path, weight in tasks_with_weights:
        # 找到当前权重最小的批次
        min_batch_idx = batch_weights.index(min(batch_weights))
        batches[min_batch_idx].append((task_name, task_path))
        batch_weights[min_batch_idx] += weight

    return batches

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
    if isinstance(raw_dir, str):
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
    
    #采用多进程，对数据进行处理
    max_workers = min(96, len(task_name_list))
    task_batches = batch_process_tasks(list(task_name_list.items()), max_workers,task_file_list)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for batch in task_batches:
            futures = []
            for task_name, task_path in batch:
                hdf5_files = task_file_list[task_name]
                futures.append(executor.submit(populate_dataset, dataset, hdf5_files, task=task_name, episodes=episodes))
            
            for future in as_completed(futures):
                try:
                    dataset = future.result()  # 更新dataset
                except Exception as e:
                    print(f"Task processing failed with error: {e}")

    dataset.consolidate()

if __name__ == "__main__":
    tyro.cli(port_aloha)