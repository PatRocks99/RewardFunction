import os
import rclpy
import numpy as np
import math
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions

def calculate_reward(observation, action):
    # Implement logic to calculate the reward based on the observation and action
    reward = 0.0  # Initalize the reward
    # action space
    speed, steering_angle = action  # split the action space
    reward += speed**2 - 1.5  # calaulate reward based on speed
    reward += math.exp(-(steering_angle**2))  # calculate reward based on steering angle

    # observation space
    # Lidar goes right to left
    # sections (Left, Right), (Front Left, Front Right)
    right = observation[0:360]
    front_right = observation[360:540]
    front_left = observation[540:720]
    left = observation[720:]
    reward += process_side(left)
    reward += process_side(right)
    reward += process_front(front_left)
    reward += process_front(front_right)

    return reward

def process_front(data):
    # clean the data by removing 0s and finding the minimum value used to calculate reward
    minimum = remove_zeros_and_find_min(data)
    if minimum > 0.4:
        return 0
    elif minimum > 0.19:
        reward = -1 / (5.7 * minimum - 1)
    else:
        # crash
        return -50
    return reward/2

def process_side(data):
    #clean the data by removing 0s and finding the minimum value used to calculate reward
    minimum = remove_zeros_and_find_min(data)
    if minimum >0.35:
        return 0
    elif minimum > 0.19:
        reward = (-1/(5.7*minimum-1))
        
    else:
        #crash
        return -50
    return reward/2

def remove_zeros_and_find_min(arr):
    # Remove 0s from the array
    arr_no_zeros = [x for x in arr if x != 0]

    # Check if the array is empty after removing zeros
    if not arr_no_zeros:
        return 1  # Return None if no elements are left

    # Find the minimum value in the modified array
    min_value = min(arr_no_zeros) 

    return min_value

def process_bag(bag_path: str):
    rclpy.init()
    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}
    
    required_topics = [
        "/odom",
        "/drive",
        "/picoScan_23460001/scan/all_segments_echo0"
    ]
    msg_types = {
        topic: get_message(type_map[topic])
        for topic in required_topics if topic in type_map
    }

    messages_by_topic = {topic: [] for topic in required_topics}

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic in messages_by_topic:
            msg = deserialize_message(data, msg_types[topic])
            messages_by_topic[topic].append((t, msg))

    def find_closest(ts, msgs):
        return min(msgs, key=lambda x: abs(x[0] - ts), default=(None, None))[1]

    drive_msgs = messages_by_topic["/drive"]
    odom_msgs = messages_by_topic["/odom"]
    scan_msgs = messages_by_topic["/picoScan_23460001/scan/all_segments_echo0"]

    observations, actions, rewards, next_observations, terminals = [], [], [], [], []

    for i in range(len(drive_msgs) - 1):
        ts, action_msg = drive_msgs[i]
        odom = find_closest(ts, odom_msgs)
        scan = find_closest(ts, scan_msgs)
        next_odom = find_closest(drive_msgs[i + 1][0], odom_msgs)
        next_scan = find_closest(drive_msgs[i + 1][0], scan_msgs)

        if not all([odom, scan, next_odom, next_scan, action_msg]):
            continue

        try:
            obs = np.concatenate([
            np.array([odom.pose.pose.position.x, odom.pose.pose.position.y]),
            np.array(scan.ranges)
            ])
            next_obs = np.concatenate([
                np.array([next_odom.pose.pose.position.x, next_odom.pose.pose.position.y]),
                np.array(next_scan.ranges)
            ])
            act = np.array([action_msg.steering_angle, action_msg.speed])
        except Exception:
            continue  # Skip malformed messages

        reward = calculate_reward(obs, act)

        observations.append(obs)
        actions.append(act)
        rewards.append(reward)
        next_observations.append(next_obs)
        terminals.append(False)

    rclpy.shutdown()

    return {
        "observations": np.array(observations),
        "actions": np.array(actions),
        "rewards": np.array(rewards),
        "next_observations": np.array(next_observations),
        "terminals": np.array(terminals)
    }

def process_all_bags(raw_dir="Camer_Car_raw", output_dir="Training"):
    os.makedirs(output_dir, exist_ok=True)

    for subdir in os.listdir(raw_dir):
        bag_folder = os.path.join(raw_dir, subdir)
        if not os.path.isdir(bag_folder):
            continue

        if not any(fname.endswith(".db3") for fname in os.listdir(bag_folder)):
            continue  # Skip folders without .db3

        print(f"📦 Processing bag: {subdir}")
        try:
            dataset = process_bag(bag_folder)
            output_path = os.path.join(output_dir, f"{subdir}.npz")
            np.savez(output_path, **dataset)
            print(f"✅ Saved: {output_path}")
        except Exception as e:
            print(f"❌ Failed to process {subdir}: {e}")

if __name__ == "__main__":
    process_all_bags("Camer_Car_raw", "Training")
