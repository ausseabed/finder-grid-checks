from setuptools import setup, find_packages

setup(
    name='ausseabed.findergc',
    namespace_packages=['ausseabed'],
    version='0.0.1',
    url='https://github.com/ausseabed/finder-grid-checks',
    author=(
        "Lachlan Hurst;"
    ),
    author_email=(
        "lachlan.hurst@gmail.com;"
    ),
    description=(
        'Quality Assurance checks for grid data derived from Multi Beam Echo '
        'Sounder data'
    ),
    entry_points={
        "gui_scripts": [],
        "console_scripts": [
            'findergc = ausseabed.findergc.app.cli:cli',
        ],
    },
    packages=[
        'ausseabed.findergc',
        'ausseabed.findergc.app',
        'ausseabed.findergc.lib',
        'ausseabed.findergc.qax'
    ],
    zip_safe=False,
    package_data={},
    install_requires=[
        'Click',
        'ausseabed.qajson',
        'ausseabed.mbesgc'
    ],
    tests_require=['pytest'],
)
