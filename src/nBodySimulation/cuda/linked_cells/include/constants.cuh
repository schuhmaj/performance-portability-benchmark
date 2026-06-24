namespace ppb {
    extern __constant__ size_t numParticles;

    extern __constant__ int x_dim;

    extern __constant__ int y_dim;

    extern __constant__ int z_dim;

    extern __constant__ int offsets[27];

    extern __constant__ int offsets_colored[8]; 

    extern __constant__ float deltaT; 
    
    extern __constant__ float cutoff_radius;

    extern __constant__ float cell_size;

    extern __constant__ float boxMin[3];

    extern __constant__ float globalForce[3];
} // namespace ppb