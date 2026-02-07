from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'g1_locomotion_policy'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'models'), glob('models/*.pt')),
    ],
    install_requires=['setuptools', 'torch', 'numpy'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your@email.com',
    description='G1 humanoid locomotion policy inference for ROS2',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'policy_node = g1_locomotion_policy.ros2_policy_node:main',
        ],
    },
)
