from setuptools import find_packages, setup
from glob import glob

package_name = 'me133a_final'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/urdf', glob('urdf/*')),
        ('share/' + package_name + '/meshes', glob('meshes/*')),
        ('share/' + package_name + '/meshes_unplugged', glob('meshes_unplugged/*')),
        ('share/' + package_name + '/rviz', glob('rviz/*')),
        ('share/' + package_name + '/launch', glob('launch/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='apasco',
    maintainer_email='apasco@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'simple_traj = me133a_final.simple_traj:main',
            'simple_traj_arms = me133a_final.simple_traj_arms:main',
            'arms_in = me133a_final.arms_in:main',
            'moving_atlas = me133a_final.moving_atlas:main',
            'bar = me133a_final.bar:main',
        ],
    },
)
