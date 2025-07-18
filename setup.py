from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='option_pricer',
    ext_modules=[
        CUDAExtension(
            name='option_pricer',
            sources=[
                'src/bindings.cpp',
                'src/option_pricer.cu',
            ],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']}
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
