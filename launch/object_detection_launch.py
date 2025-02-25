
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='object_detection',
            executable='object_detector',
            name='object_detector',
            output='screen',
            parameters=[],
        ),
    ])
