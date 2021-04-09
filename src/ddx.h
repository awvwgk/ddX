#ifndef DDX_H
#define DDX_H

/** C header for interfacing with ddx */

#ifdef __cplusplus
extern "C" {
#endif

// TODO const qualifiers to pointers

int ddx_get_supported_lebedev_grids(int n, int* grids);

// pass solver arguments delayed
// no print level in this function, just problem setup and discretisation, fmm
void* ddx_init(int nsph, const double* charge, const double* x, const double* y,
               const double* z, const double* rvdw, int model, int lmax, int ngrid,
               int force, int fmm, int pm, int pl, double se, double eta, double eps,
               double kappa, int nproc, int* info);
void ddx_finish(void* ddx);

int ddx_get_ncav(const void* ddx);
int ddx_get_nsph(const void* ddx);
int ddx_get_nbasis(const void* ddx);
int ddx_get_ngrid(const void* ddx);
int ddx_get_lmax(const void* ddx);
double ddx_get_epsilon(const void* ddx);

// get cavity coordinates, column-major array (3, ncav)
void ddx_get_ccav(const void* ddx, int ncav, double* ccav);

// Get scaled ylm at a point and with respect to a cavity sphere
void ddx_scaled_ylm(const void* c_ddx, int lmax, const double* x, int sphere,
                    double* ylm);

// This is a misnomer ... computes the classical electrostatics contribution
// from the atoms stored inside ddx using the ddx_init.
//
// phi: electrostatic potential at the cavity points  (what people call MEP)
// psi: multipolar representation of potential from solute charge inside a cavity sphere
//
void ddx_mkrhs(const void* ddx, int nsph, int ncav, int nbasis, double* phi_cav,
               double* gradphi_cav, double* psi);

void ddx_mkxi(const void* ddx, int nsph, int ncav, int nbasis, const double* s,
              double* xi);

// phi is only needed to solve the response system
// psi is only needed for the energy

// Separate functions for energy and forces !
// i.e. one function to solve the dd system and return X
// one to take X and compute the energy
// one to take X, \phi, ... and compute the force
//
void ddx_solve(const void* ddx, int nsph, int ncav, int nbasis, const double* phi_cav,
               const double* gradphi_cav, const double* psi, int itersolver, double tol,
               int maxiter, int ndiis, int iprint, double* esolv, double* xs, double* s,
               double* force);

#ifdef __cplusplus
}
#endif

#endif /* DDX_H */
