#pragma once

namespace ppb::cuda::nbody {
    /**
    * The number of particles in the simulation domain 
    */
    __constant__ inline size_t NUM_PARTICLES;
    
    /**
    * The size of one timestep
    */ 
    __constant__ inline float DELTA_T;
   
    /**
    * The cutoff radius
    */
    __constant__ inline float CUTOFF_RADIUS;

    /**
    * The cutoff radius squared
    */
    __constant__ inline float CUTOFF_RADIUS_SQUARED;

    /**
    * The Verlet skin
    */
    __constant__ inline float VERLET_SKIN;

    /**
    * The global force to be applied to each particle during each simulation step. E.g. (0, 0, 0), (0, -9.81, 0), ...
    */
    __constant__ inline float GLOBAL_FORCE[3];

    /**
    * The smallest x, y, and z coordinate in the simulation domain
    */
    __constant__ inline float BOX_MIN[3];

    /**
    * The largest x, y, and z coordinate in the simulation domain
    */
    __constant__ inline float BOX_MAX[3];
    
    /**
    * [LC OPT] The number of cells in the x-dimension of the simulation domain
    */
    __constant__ inline int X_DIM;
    
    /**
    * [LC OPT] The number of cells in the y-dimension of the simulation domain
    */ 
    __constant__ inline int Y_DIM;
    
    /**
    * [LC OPT] The number of cells in the z-dimension of the simulation domain
    */
    __constant__ inline int Z_DIM;
    
    /**
    * [LC OPT] The *index* offsets of each of the 27 cell neighbors of a given cell
    */ 
    __constant__ inline int OFFSETS[27];
    
    /**
    * [LC OPT] The offsets in (x,y,z)-tuples of each of the 27 cell neighbors of a given cell
    */ 
    __constant__ inline int OFFSETS_XYZ[81];

    /**
    * [LC OPT] The derived cell size (= CUTOFF_RADIUS + VERLET_SKIN)
    */
    __constant__ inline float CELL_SIZE;
} // namespace ppb::cuda::nbody