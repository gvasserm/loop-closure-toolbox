from mcap.mcap0.stream_reader import StreamReader

#stream = open("/home/gvasserm/dev/aicv_amr_ws/Warehouse10K_2_4_D455f_Cameras_20240324_114618849/Warehouse10K_2_4_D455f_Cameras_20240324_114618849_0.mcap", "rb")
import cv2
import rosbag2_py
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pathlib import Path
import argparse
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
import zstandard as zstd
from tqdm import tqdm
import os
import json

"""
Usage
=====
pip install zstandard
python --bag <path_to_mcap_or_db3_bag> --topic <image sequence topic name> --fps 30 --flip --output /tmp/vid.mp4

note that python should be python with ros packages, it might not be your current conda env python, e.g. use /usr/bin/python3
installing dependencies should also use the pip with ros packages, for example `/usr/bin/pip3 install zstandard`
"""


zstd_decompressor = zstd.ZstdDecompressor()


def get_reader(input_bag: Path):
    """
    Refer to https://mcap.dev/guides/python/ros2
    """
    storage_id = 'mcap'
    if input_bag.suffix == '.db3':
        storage_id = 'sqlite3'
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(input_bag), storage_id=storage_id),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )

    topic_types = reader.get_all_topics_and_types()

    def typename(topic_name):
        for topic_type in topic_types:
            if topic_type.name == topic_name:
                return topic_type.type
        raise ValueError(f"topic {topic_name} not in bag")

    return reader, typename


def get_decompressor(input_bag: Path, image_topic: str):
    reader, typename = get_reader(input_bag)

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        if topic != image_topic:
            continue
        break

    assert topic == image_topic, f"Could not find messages in the requested topic [{image_topic}]."
    msg_type = get_message(typename(topic))
    try:
        deserialize_message(data, msg_type)
        return None
    except:
        try:
            data_zstd = zstd_decompressor.decompress(data)
            deserialize_message(data_zstd, msg_type)
            return zstd_decompressor
        except:
            raise Exception(f"Could not parse messages from topic [{topic}]")


def read_messages(input_bag: Path, image_topic: str):
    reader, typename = get_reader(input_bag)
    decompressor = get_decompressor(input_bag, image_topic)

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        if topic != image_topic:
            continue
        msg_type = get_message(typename(topic))

        if decompressor is not None:
            data = decompressor.decompress(data)
        msg = deserialize_message(data, msg_type)
        yield topic, msg, timestamp
    del reader
    

def extract_image_sequence(bag_path: Path, image_topic: str, output_path: Path, fps: int, is_flip):
    bridge = CvBridge()
    video_writer = None
    frame_width = None
    frame_height = None

    cameraID = int(image_topic.split('camera')[-1][0])

    data_info = {}

    if str(output_path).split('.')[-1] == 'mp4':
        frame_width = cv_image.shape[1]
        frame_height = cv_image.shape[0]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
        output_dir = output_path.parent
    else:
        fname = f"cam_{cameraID}_frame_"
        output_dir = output_path

    for frameID, (topic, msg, timestamp) in enumerate(tqdm(read_messages(bag_path, image_topic))):
        if topic != image_topic:
            continue
        assert isinstance(msg, Image)

        data_info[frameID] = timestamp
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if is_flip:
            cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)
        if video_writer is None:
            fname_ = fname + str(frameID).zfill(5) + ".png"
            cv2.imwrite(os.path.join(output_dir, fname_), cv_image)
        else:
            video_writer.write(cv_image)
    if video_writer is not None:
        video_writer.release()
        print("Done.")
        print(f"Saved video to path [{output_path}].")
    else:
        print("Could not generate video from topic, are you sure the topic exists in the bag file?")

    with open(output_dir / f"data_cam{cameraID}.json", "w") as f:
        json.dump(data_info, f)


def main():
    parser = argparse.ArgumentParser(description="Extract image sequence from ROS 2 bag to MP4 video.")
    parser.add_argument("--bag", help="Path to ROS2 bag file (.mcap or .db3 file)")
    parser.add_argument("--topic", help="Image topic in the bag file")
    parser.add_argument("--output", help="Path to MP4 video output")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--flip", action="store_true", help="Flip the frames 180 deg")
    args = parser.parse_args()
    bag_path = Path(args.bag).expanduser()
    output_path = Path(args.output).expanduser()
    #assert output_path.suffix == '.mp4', "The only supported video format is mp4"
    assert bag_path.is_file() or bag_path.is_dir()
    extract_image_sequence(bag_path, args.topic, output_path, args.fps, args.flip)


if __name__ == '__main__':
    main()