from setuptools import setup


package_name = "sim"


setup(
    name=package_name,
    version="0.0.1",
    packages=["sim_stereo"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/sim"]),
        ("share/sim", ["package.xml"]),
        ("share/sim/launch", ["launch/stereo_render.launch.py"]),
        ("share/sim/config", ["config/defaults.yaml"]),
        ("share/sim", ["README.md", "NEW_PACKAGES.md"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="hc",
    maintainer_email="hc@localhost",
    description="Real-time stereo mesh renderer and ROS2 image publisher.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "stereo_render_node = sim_stereo.node:main",
        ],
    },
)
