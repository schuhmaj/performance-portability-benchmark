#pragma once

namespace ppb::cuda::nbody {
    /**
    * The number of particles in the simulation domain 
    */
    __constant__ inline size_t NUM_PARTICLES;

    /**
    * The number of cells in the x-dimension of the simulation domain
    */
    __constant__ inline int X_DIM;

    /**
    * The number of cells in the y-dimension of the simulation domain
    */
    __constant__ inline int Y_DIM;

    /**
    * The number of cells in the z-dimension of the simulation domain
    */
    __constant__ inline int Z_DIM;

    /**
    * The *index* offsets of each of the 27 cell neighbors of a given cell
    */
    __constant__ inline int OFFSETS[27];

    /**
    * The offsets in (x,y,z)-tuples of each of the 27 cell neighbors of a given cell
    */
    __constant__ inline int OFFSETS_XYZ[81];

    /**
    * [DOMAIN COLORING]
    * The indices of the offsets in OFFSETS of each of the 8 cell neighbors of a given cell
    */
    __constant__ inline int OFFSETS_COLORED[8]; 
   
    /**
    * [DOMAIN COLORING]
    * The indices of the offsets in OFFSETS of each of the pairs of cell neighbors of a given cell
    */
    __constant__ inline int OFFSETS_COLORED_NON_BASE_CELL[12]; 

    /**
    * [DOMAIN COLORING]
    * The number of cells that have the same color in the simulation domain
    */
    __constant__ inline int NUM_CELLS_SAME_COLOR;

    /**
    * [DOMAIN COLORING]
    * X_DIM rounded up to the nearest multiple of 4
    */
    __constant__ inline int X_DIM_NEAREST_4;
    
    /**
    * [DOMAIN COLORING]
    * Y_DIM rounded up to the nearest multiple of 4
    */
    __constant__ inline int Y_DIM_NEAREST_4;
    
    /**
    * [DOMAIN COLORING]
    * Z_DIM rounded up to the nearest multiple of 4
    */
    __constant__ inline int Z_DIM_NEAREST_4;

    /**
    * The size of one timestep
    */
    __constant__ inline float DELTA_T; 
    
    /**
    * The derived cutoff radius (= CELL_SIZE) squared
    */
    __constant__ inline float CUTOFF_RADIUS_SQUARED;

    /**
    * The side length of one cell. Each cell is a *cube*, i.e. same side length for each side.
    */
    __constant__ inline float CELL_SIZE;

    /**
    * The smallest x, y, and z coordinate in the simulation domain
    */
    __constant__ inline float BOX_MIN[3];

    /**
    * The global force to be applied to each particle during each simulation step. E.g. (0, 0, 0), (0, -9.81, 0), ...
    */
    __constant__ inline float GLOBAL_FORCE[3];
} // namespace ppb::cuda::nbody