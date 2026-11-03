from glob import glob

from setuptools import setup

setup(
    name="gdnav_ros",
    version="0.3.0",
    packages=["gdnav_ros"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/gdnav_ros"]),
        ("share/gdnav_ros", ["package.xml"]),
        ("share/gdnav_ros/launch", glob("launch/*.py")),
        ("share/gdnav_ros/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    entry_points={"console_scripts": [
        "mission_node = gdnav_ros.mission_node:main",
        "vio_bridge_node = gdnav_ros.vio_bridge_node:main",
    ]},
)
