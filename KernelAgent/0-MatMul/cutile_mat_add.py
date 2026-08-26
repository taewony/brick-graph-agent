from math import ceil
import cupy as cp
import numpy as np
import cuda.tile as ct


@ct.kernel
def matrix_add(a, b, c,
               tile_size: ct.Constant[int],
               num_tiles_col: ct.Constant[int]):

    # ------------------------------------------------------------
    # 1. Linear tile id
    # ------------------------------------------------------------
    pid = ct.bid(0)

    tile_row = pid // num_tiles_col
    tile_col = pid %  num_tiles_col

    # ------------------------------------------------------------
    # 2. Load tiles
    #    ⚠️ index는 "타일 좌표"다!
    # ------------------------------------------------------------
    a_tile = ct.load(
        a,
        index=(tile_row, tile_col),
        shape=(tile_size, tile_size)
    )

    b_tile = ct.load(
        b,
        index=(tile_row, tile_col),
        shape=(tile_size, tile_size)
    )

    # ------------------------------------------------------------
    # 3. Compute
    # ------------------------------------------------------------
    c_tile = a_tile + b_tile

    # ------------------------------------------------------------
    # 4. Store
    # ------------------------------------------------------------
    ct.store(
        c,
        index=(tile_row, tile_col),
        tile=c_tile
    )


def test():
    matrix_size = 2**12   # 4096
    tile_size   = 2**5    # 32

    assert matrix_size % tile_size == 0

    num_tiles = matrix_size // tile_size
    grid = (num_tiles * num_tiles, 1, 1)

    a = cp.random.uniform(-1, 1, (matrix_size, matrix_size)).astype(np.float32)
    b = cp.random.uniform(-1, 1, (matrix_size, matrix_size)).astype(np.float32)
    c = cp.zeros_like(a)

    ct.launch(
        cp.cuda.get_current_stream(),
        grid,
        matrix_add,
        (a, b, c, tile_size, num_tiles)
    )

    np.testing.assert_allclose(
        cp.asnumpy(c),
        cp.asnumpy(a + b),
        rtol=1e-6
    )

    print("✓ matrix_add passed correctly!")


if __name__ == "__main__":
    test()

