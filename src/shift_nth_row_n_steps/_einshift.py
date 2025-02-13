from ivy import Array, NativeArray


def einshift(order: str, indexing: str, a: Array | NativeArray) -> Array:
    # axis order: ijk, indexing: i-j,k
    # i,k ->_repeat i,i,k -> shift
    pass


# (i,j) -> (i,ai+j) forall a in Z.
# (i,j) -> (ai,j) forall a in Z.
# (ai,j) -> (i,j) forall a in Z.

# (i,j) -> (i,aj) forall a in Z.
# (i,j) -> (i+aj,j) forall a in Z.
# (i,aj) -> (i,j) forall a in Z.

# (i,j) -> (ai+bj,ci+dj) can be expressed as
# (i,j) ->
# -> (ac i, bdj)
# -> (acdi+bcdj, abdj)
# -> (acdi+bcdj,abci+abdj)
