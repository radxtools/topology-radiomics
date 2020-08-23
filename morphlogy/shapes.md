    mri_voxels          : X x Y x Z
    smoothed_mri_voxels : X x Y x Z
    verts               : Vx3
    faces               : Fx3 
    dim                 : int . ceil . sqrt(V)
    N                   : dim * dim
    V1                  : N x 3
    copy V to V1
    assert N  >= V
    reshape V1 for each dimension in i,j,k to get XX,YY,ZZ
    XX                  : dim x dim
    YY                  : dim x dim
    ZZ                  : dim x dim
    curvature.*         : dim x dim
    surface_measures.*  : dim x dim
    intensity = curvature.*.flatten() | surface_measures.*.flatten()
    intensity           : 1 x N


Should get a table like the following
where:   
      X is the x-coordinate of the vertex  
      Y is the y-coordinate of the vertex  
      Z is the z-coordinate of the vertex  

|            | X | Y | Z | Intensity      |
|------------|---|---|---|----------------|
|    V1[0]   | 1 | 1 | 1 | intensity[0]   |
|    V1[1]   | 1 | 1 | 1 | intensity[1]   |
|     .      | . | . | . | .              |
|     .      | . | . | . | .              |
|     .      | . | . | . | .              |
|    V1[N-1] | 1 | 1 | 1 | intensity[N-1] |