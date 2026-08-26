"""
Hello cuTile!
Demonstrates using ct.printf inside a cuTile kernel.
"""

import cuda.tile as ct
import cupy as cp


# ------------------------------------------------------------
# cuTile kernel
# ------------------------------------------------------------
@ct.kernel
def hello_cutile_kernel():
    """
    A minimal cuTile kernel that prints a message from the GPU.
    """

    # Print from device code
    ct.printf("hello cutile\n")


# ------------------------------------------------------------
# Host-side code
# ------------------------------------------------------------
def main():
    # ------------------------------------------------------------
    # IMPORTANT: Initialize CUDA context by touching the GPU
    # ------------------------------------------------------------
    cp.zeros(1)   # <-- THIS LINE FIXES THE ERROR
    
    # Launch the kernel
    # Only one tile / one program is enough
    ct.launch(
        cp.cuda.get_current_stream(),
        (1, 1, 1),
        hello_cutile_kernel,
        ()
    )

    # Synchronize to make sure printf output is flushed
    cp.cuda.get_current_stream().synchronize()

    print("Kernel launched successfully.")


if __name__ == "__main__":
    main()

