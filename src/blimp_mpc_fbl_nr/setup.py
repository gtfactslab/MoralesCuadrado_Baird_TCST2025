from setuptools import setup

package_name = 'blimp_mpc_fbl_nr'

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
    maintainer='factslabegmc',
    maintainer_email='evannsmcuadrado@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'run_blimp_sim = blimp_mpc_fbl_nr.run_blimp_sim:main',

            'run_wardi_circle_horz = blimp_mpc_fbl_nr.x1_official_run_wardi_circle_horz:main',
            'run_wardi_circle_vert = blimp_mpc_fbl_nr.x2_official_run_wardi_circle_vert:main',
            'run_wardi_fig8_horz = blimp_mpc_fbl_nr.x3_official_run_wardi_fig8_horz:main',
            'run_wardi_fig8_vert_short = blimp_mpc_fbl_nr.x4_official_run_wardi_fig8_vert_short:main',
            'run_wardi_fig8_vert_tall = blimp_mpc_fbl_nr.x5_official_run_wardi_fig8_vert_tall:main',
            'run_wardi_circle_horz_spin = blimp_mpc_fbl_nr.x6_official_run_wardi_circle_horz_spin:main',
            'run_wardi_helix = blimp_mpc_fbl_nr.x7_official_run_wardi_helix:main',
            'run_wardi_helix_spin = blimp_mpc_fbl_nr.x8_official_run_wardi_helix_spin:main',


            'run_nlmpc_circle_horz = blimp_mpc_fbl_nr.y1_official_run_nlmpc_circle_horz:main',
            'run_nlmpc_circle_vert = blimp_mpc_fbl_nr.y2_official_run_nlmpc_circle_vert:main',
            'run_nlmpc_fig8_horz = blimp_mpc_fbl_nr.y3_official_run_nlmpc_fig8_horz:main',
            'run_nlmpc_fig8_vert_short = blimp_mpc_fbl_nr.y4_official_run_nlmpc_fig8_vert_short:main',
            'run_nlmpc_fig8_vert_tall = blimp_mpc_fbl_nr.y5_official_run_nlmpc_fig8_vert_tall:main',
            'run_nlmpc_circle_horz_spin = blimp_mpc_fbl_nr.y6_official_run_nlmpc_circle_horz_spin:main',
            'run_nlmpc_helix = blimp_mpc_fbl_nr.y7_official_run_nlmpc_helix:main',
            'run_nlmpc_helix_spin = blimp_mpc_fbl_nr.y8_official_run_nlmpc_helix_spin:main',

            'run_cbf_circle_horz = blimp_mpc_fbl_nr.z1_official_run_cbf_circle_horz:main',
            'run_cbf_circle_vert = blimp_mpc_fbl_nr.z2_official_run_cbf_circle_vert:main',
            'run_cbf_fig8_horz = blimp_mpc_fbl_nr.z3_official_run_cbf_fig8_horz:main',
            'run_cbf_fig8_vert_short = blimp_mpc_fbl_nr.z4_official_run_cbf_fig8_vert_short:main',
            'run_cbf_fig8_vert_tall = blimp_mpc_fbl_nr.z5_official_run_cbf_fig8_vert_tall:main',
            'run_cbf_circle_horz_spin = blimp_mpc_fbl_nr.z6_official_run_cbf_circle_horz_spin:main',
            'run_cbf_helix = blimp_mpc_fbl_nr.z7_official_run_cbf_helix:main',
            'run_cbf_helix_spin = blimp_mpc_fbl_nr.z8_official_run_cbf_helix_spin:main',

        ],
    },
)
