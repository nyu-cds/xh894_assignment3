"""
    N-body simulation.
    
"""
'''
    Version_numba: Final version
    Version_opt execution time: 36.08s
    Version_numba execution time :6.98s
    Relative Speedup : 5.169
   
import itertools
import numpy as np
from numba import jit, int64, float64, void, vectorize
PI = 3.14159265358979323
SOLAR_MASS = 4 * PI * PI
DAYS_PER_YEAR = 365.24
BODIES = np.array([
    ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [SOLAR_MASS,0.0,0.0]),

    ([4.84143144246472090e+00,
                 -1.16032004402742839e+00,
                 -1.03622044471123109e-01],
                [1.66007664274403694e-03 * DAYS_PER_YEAR,
                 7.69901118419740425e-03 * DAYS_PER_YEAR,
                 -6.90460016972063023e-05 * DAYS_PER_YEAR],
                [9.54791938424326609e-04 * SOLAR_MASS,0.0,0.0]),

   ([8.34336671824457987e+00,
                4.12479856412430479e+00,
                -4.03523417114321381e-01],
               [-2.76742510726862411e-03 * DAYS_PER_YEAR,
                4.99852801234917238e-03 * DAYS_PER_YEAR,
                2.30417297573763929e-05 * DAYS_PER_YEAR],
               [2.85885980666130812e-04 * SOLAR_MASS,0.0,0.0]),

     ([1.28943695621391310e+01,
                -1.51111514016986312e+01,
                -2.23307578892655734e-01],
               [2.96460137564761618e-03 * DAYS_PER_YEAR,
                2.37847173959480950e-03 * DAYS_PER_YEAR,
                -2.96589568540237556e-05 * DAYS_PER_YEAR],
               [4.36624404335156298e-05 * SOLAR_MASS,0.0,0.0]),

    ([1.53796971148509165e+01,
                 -2.59193146099879641e+01,
                 1.79258772950371181e-01],
                [2.68067772490389322e-03 * DAYS_PER_YEAR,
                 1.62824170038242295e-03 * DAYS_PER_YEAR,
                 -9.51592254519715870e-05 * DAYS_PER_YEAR],
                [5.15138902046611451e-05 * SOLAR_MASS,0.0,0.0])])
    #generate pairs
uniq_body_pair = np.array(list(itertools.combinations(np.arange(BODIES.shape[0]), 2)))
bodies_name = {'sun': 0,'jupiter': 1,'saturn': 2,'uranus': 3,'neptune': 4}

@vectorize([float64(float64, float64)])
def vec_deltas(x, y):
    return x - y

@jit('void(float64, int64,float64[:,:,:])',nopython=True)
def advance(dt,iterations,bodies):
    '''
        advance the system one timestep
    '''
    for _ in range(iterations):
        for index in range(len(uniq_body_pair)):
        #update
            (body1,body2) = uniq_body_pair[index]
            p1 = bodies[body1][0]
            v1 = bodies[body1][1]
            m1 = bodies[body1][2][0]
            p2 = bodies[body2][0]
            v2 = bodies[body2][1]
            m2 = bodies[body2][2][0]
            (dx, dy, dz) = vec_deltas(p1, p2)    
            temp = dt * ((dx * dx + dy * dy + dz * dz) ** (-1.5)) 
            b1 = temp * m1
            b2 = temp * m2
            v1[0] -= dx * b2
            v1[1] -= dy * b2
            v1[2] -= dz * b2
            v2[0] += dx * b1
            v2[1] += dy * b1
            v2[2] += dz * b1
        for index in range(len(bodies)):
        
            r = bodies[index][0]
            v = bodies[index][1]
            r += dt * v
            

@jit("float64(float64,float64[:,:,:])",nopython=True)  
def report_energy( e, bodies):
    '''
        compute the energy and return it so that it can be printed
    '''
    for index in range(len(uniq_body_pair)):
        (body1,body2) = uniq_body_pair[index]
        p1 = bodies[body1][0]
        v1 = bodies[body1][1]
        m1 = bodies[body1][2][0]
        p2 = bodies[body2][0]
        v2 = bodies[body2][1]
        m2 = bodies[body2][2][0]
        (dx, dy, dz) = vec_deltas(p1, p2)
        e -= (m1 * m2) / ((dx * dx + dy * dy + dz * dz) ** 0.5)
    for index in range(len(bodies)):    
        v = bodies[index][1]
        m = bodies[index][2][0]
        e += m * (np.sum(v**2)) / 2.
    return e




@jit('void(int64, int64, int64, float64[:,:,:])',nopython=True)
def nbody(loops, reference, iterations, bodies):
    '''
        nbody simulation
        loops - number of loops to run
        reference - body at center of system
        iterations - number of timesteps to advance
    '''
    p = np.array([0.0,0.0,0.0])
    for index in range(len(bodies)): 
        r = bodies[index][0]
        v = bodies[index][1]
        m = bodies[index][2][0]
        p -= v * m
    v = bodies[reference][1]
    m = bodies[reference][2][0]
    v = p / m
    for _ in range(loops):
        advance(0.01,iterations,bodies)
        print(report_energy(0.0, bodies))

if __name__ == '__main__':
    import timeit
    print(timeit.timeit("nbody(100, bodies_name['sun'], 20000, BODIES)", setup="from __main__ import nbody, bodies_name, BODIES", number=1))
