from setuptools import setup

package_name = 'quad_newton_raphson_flow'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Evanns Morales-Cuadrado',
    maintainer_email='egm@gatech.edu',
    description='Newton Raphson Controller for Drone Applications in Simulation and Hardware Using PX4 Flight Stack',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'nr_quad = quad_newton_raphson_flow.nr_flow_quad:main',
        ],
    },
)
