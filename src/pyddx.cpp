#include "ddx.h"
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <string>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
using namespace pybind11::literals;
using array_f_t = py::array_t<double, py::array::f_style | py::array::forcecast>;

void export_pyddx_data(py::module& m);  // defined in pyddx_data.cpp

// Representation of the solvation contribution computed by the ddX library
struct Solvation {
  double energy;
  array_f_t X;   // The forward DD-COSMO / PCM / LPB solution
  array_f_t S;   // The adjoint DD-COSMO / PCM / LPB solution
  array_f_t xi;  // The charges on the DD-COSMO atom-wise Lebedev grid (per atom).
  array_f_t force;

  Solvation(int n_spheres, int n_basis, int n_cav)
        : energy(0.0),
          X({n_basis, n_spheres}),
          S({n_basis, n_spheres}),
          xi({n_cav}),
          force({3, n_spheres}) {
    std::fill(force.mutable_data(), force.mutable_data() + force.size(), 0.0);
    std::fill(X.mutable_data(), X.mutable_data() + X.size(), 0.0);
    std::fill(S.mutable_data(), S.mutable_data() + S.size(), 0.0);
    std::fill(xi.mutable_data(), xi.mutable_data() + xi.size(), 0.0);
  }
};

class Model {
 public:
  Model(std::string model, array_f_t sphere_charges, array_f_t sphere_centres,
        array_f_t sphere_radii, double solvent_epsilon, double solvent_kappa, int lmax,
        int n_lebedev, double eta, bool use_fmm, int fmm_multipole_lmax,
        int fmm_local_lmax)
        : m_holder(nullptr), m_model(model) {
    int model_id = 0;
    if (model == "cosmo") {
      model_id = 1;
    } else if (model == "pcm") {
      model_id = 2;
    } else {
      throw py::value_error("Invalid model string: " + model);
    }

    // Check size of vdW and atomic data
    const size_t n_spheres = sphere_charges.size();
    if (sphere_charges.ndim() != 1) {
      throw py::value_error("Parameter sphere_charges is not a 1D array.");
    }
    if (sphere_radii.ndim() != 1) {
      throw py::value_error("Parameter sphere_radii is not a 1D array.");
    }
    if (n_spheres != sphere_radii.size()) {
      throw py::value_error("Length of 'sphere_charges' and 'sphere_radii' don't agree.");
    }
    if (sphere_centres.ndim() != 2) {
      throw py::value_error("sphere_centres is not a 2D array.");
    }
    if (3 != sphere_centres.shape(1) || n_spheres != sphere_centres.shape(0)) {
      throw py::value_error("sphere_centres should be a n_spheres x 3 array.");
    }

    // Get supported Lebedev grids
    std::vector<int> supported_grids(100);
    const int n_supp_grids =
          ddx_get_supported_lebedev_grids(supported_grids.size(), supported_grids.data());
    supported_grids.resize(static_cast<size_t>(n_supp_grids));
    if (std::find(supported_grids.begin(), supported_grids.end(), n_lebedev) ==
        supported_grids.end()) {
      std::string msg = "Lebedev grid size '" + std::to_string(n_lebedev) +
                        "' not supported. Supported grid sizes are: ";
      for (size_t i = 0; i < supported_grids.size(); ++i) {
        const std::string separator = i < supported_grids.size() ? ", " : "";
        msg += std::to_string(supported_grids[i]) + separator;
      }
      throw py::value_error(msg);
    }

    if (eta < 0 || eta > 1) {
      throw py::value_error("Regularisation parameter eta needs to be between 0 and 1.");
    }
    if (solvent_epsilon <= 0) {
      throw py::value_error(
            "Dielectric permitivity 'solvent_epsilon' needs to be positive.");
    }
    if (lmax < 0) {
      throw py::value_error("Maximal spherical harmonics degree 'lmax' needs to be >= 0");
    }
    if (fmm_multipole_lmax < -1) {
      throw py::value_error(
            "Maximal spherical harmonics degree 'fmm_multipole_lmax' needs to >= -1 with "
            "-1 disabling far-field FMM contributions.");
    }
    if (fmm_local_lmax < -1) {
      throw py::value_error(
            "Maximal spherical harmonics degree 'fmm_local_lmax' needs to >= -1 with -1 "
            "disabling local FMM contributions.");
    }

    const double se = 0.0;  // Hard-code centred regularisation
    const int force = 1;    // Always support force calculations.
    const int fmm   = use_fmm ? 1 : 0;
    const int nproc = 1;  // For now no parallelisation is supported
    int info        = 0;
    m_holder = ddx_init(n_spheres, sphere_charges.data(), sphere_centres.data(0, 0),
                        sphere_centres.data(0, 1), sphere_centres.data(0, 2),
                        sphere_radii.data(), model_id, lmax, n_lebedev, force, fmm,
                        fmm_multipole_lmax, fmm_local_lmax, se, eta, solvent_epsilon,
                        solvent_kappa, nproc, &info);

    if (info != 0) {
      throw py::value_error("Info from ddx_init nonzero: " + std::to_string(info) + ".");
    }
  }

  ~Model() {
    if (m_holder) ddx_finish(m_holder);
    m_holder = nullptr;
  }
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  int n_cav() const { return ddx_get_ncav(m_holder); }
  int n_spheres() const { return ddx_get_nsph(m_holder); }
  int n_basis() const { return ddx_get_nbasis(m_holder); }
  int n_lebedev() const { return ddx_get_ngrid(m_holder); }
  int lmax() const { return ddx_get_lmax(m_holder); }
  std::string model() const { return m_model; }
  double solvent_epsilon() const { return ddx_get_epsilon(m_holder); }
  bool needs_gradphi() const { return m_model == "lpb"; }

  array_f_t cavity() const {
    array_f_t coords({3, n_cav()});
    ddx_get_ccav(m_holder, n_cav(), coords.mutable_data());
    return coords;
  }

  py::array_t<double> scaled_ylm(array_f_t coord, int sphere,
                                 py::array_t<double> out) const {
    if (out.size() != n_basis()) {
      throw py::value_error("'out' should have the same size as `n_basis()`");
    }
    if (coord.size() != 3) {
      throw py::value_error("'coord' should have exactly three entries");
    }
    if (sphere >= n_spheres()) {
      throw py::value_error("'sphere' should be less than n_spheres()");
    }
    ddx_scaled_ylm(m_holder, lmax(), coord.data(), sphere + 1, out.mutable_data());
    return out;
  }
  array_f_t scaled_ylm(array_f_t coord, int sphere) const {
    return scaled_ylm(coord, sphere, array_f_t({n_basis()}));
  }

  /** Nuclear contribution to the cavity charges and potential */
  py::dict solute_nuclear_contribution() const {
    array_f_t phi({n_cav()});
    array_f_t gradphi({3, n_cav()});
    array_f_t psi({n_basis(), n_spheres()});
    ddx_mkrhs(m_holder, n_spheres(), n_cav(), n_basis(), phi.mutable_data(),
              gradphi.mutable_data(), psi.mutable_data());
    return py::dict("phi"_a = phi, "gradphi"_a = gradphi, "psi"_a = psi);
  }

  // TODO Should also have a version that takes gradphi (for LPB later)
  Solvation compute(array_f_t phi,  // Electrostatic potential at the cavity points
                    array_f_t psi,  // Multipolar representation of potential from solute
                    double tol,     // Relative error threshold for an iterative solver
                    int maxiter,  // Maximum number of iterations for an iterative solver
                    int ndiis,    // Size of the DIIS history
                    int print     // > 0
  ) {
    if (phi.ndim() != 1 || phi.shape(0) != n_cav()) {
      throw py::value_error("phi not of shape (n_cav, ) == (" + std::to_string(n_cav()) +
                            ").");
    }
    if (psi.ndim() != 2 || psi.shape(0) != n_basis() || psi.shape(1) != n_spheres()) {
      throw py::value_error("psi not of shape (n_basis, n_spheres) == (" +
                            std::to_string(n_basis()) + ", " +
                            std::to_string(n_spheres()) + ").");
    }
    /*
    if (gradphi.ndim() != 2 || gradphi.shape(0) != 3 ||
        gradphi.shape(1) != n_cav()) {
      throw py::value_error("gradphi not of shape (3, n_cav) == (3, " +
                            std::to_string(n_cav()) + ").");
    }
    */
    array_f_t gradphi({3, n_cav()});
    std::fill(gradphi.mutable_data(), gradphi.mutable_data() + gradphi.size(), 0.0);

    if (tol < 1e-14 || tol > 1) {
      throw py::value_error("Tolerance 'tol' needs to be in range [1e-14, 1]");
    }
    if (maxiter <= 0) {
      throw py::value_error("Maxiter needs to be positive.");
    }
    if (ndiis <= 0) {
      throw py::value_error("'ndiis' needs to be positive.");
    }
    if (print < 0) {
      throw py::value_error("'print' needs to be non-negative.");
    }

    const int itersolver = 1;
    Solvation ret(n_spheres(), n_basis(), n_cav());
    ddx_solve(m_holder, n_spheres(), n_cav(), n_basis(), phi.data(), gradphi.data(),
              psi.data(), itersolver, tol, maxiter, ndiis, print, &ret.energy,
              ret.X.mutable_data(), ret.S.mutable_data(), ret.force.mutable_data());
    ddx_mkxi(m_holder, n_spheres(), n_cav(), n_basis(), ret.S.data(),
             ret.xi.mutable_data());
    return ret;
  }

 private:
  void* m_holder;
  std::string m_model;
};

void export_pyddx_classes(py::module& m) {
  py::class_<Solvation, std::shared_ptr<Solvation>>(
        m, "Solvation",
        "Result when solving the solvation model for a particular solute potential.")
        .def_readonly("energy", &Solvation::energy)
        .def_readonly("X", &Solvation::X)
        .def_readonly("S", &Solvation::S)
        .def_readonly("xi", &Solvation::xi)
        .def_readonly("force", &Solvation::force);

  // TODO Better docstring
  const char* init_docstring =
        "Setup a solvation model for use with ddX\n\n"
        "sphere_charges:   n_spheres array\n"
        "atomic_centers:   (n_spheres, 3) array\n"
        "sphere_radii:     n_spheres array\n"
        "solvent_epsilon:  Relative dielectric permittivity\n"
        "solvent_kappa:    Debye-Hückel parameter (inverse screening length)\n"
        "lmax:             Maximal degree of modelling spherical harmonics\n"
        "n_lebedev:        Number of Lebedev grid points to use\n"
        "eta:              Regularization parameter\n"
        "use_fmm:          Use fast-multipole method (true) or not (false)\n"
        "fmm_multipole_lmax:  Maximal degree of multipole spherical harmonics, "
        "ignored "
        "in case `use_fmm=false`. Value `-1` means no far-field FFM interactions "
        "are "
        "computed.\n"
        "fmm_local_lmax:   Maximal degree of local spherical harmonics, ignored in "
        "case "
        "`use_fmm=false`. Value `-1` means no local FFM interactions are "
        "computed.\n";

  py::class_<Model, std::shared_ptr<Model>>(m, "Model",
                                            "Solvation model using ddX library.")
        .def(py::init<std::string, array_f_t, array_f_t, array_f_t, double, double, int,
                      int, double, bool, int, int>(),
             init_docstring, "model"_a, "charges"_a, "centres"_a, "radii"_a,
             "solvent_epsilon"_a, "solvent_kappa"_a = 0.0, "lmax"_a = 10,
             "n_lebedev"_a = 302, "eta"_a = 0.1, "use_fmm"_a = true,
             "fmm_multipole_lmax"_a = 20, "fmm_local_lmax"_a = 20)
        .def_property_readonly("n_cav", &Model::n_cav)
        .def_property_readonly("n_spheres", &Model::n_spheres)
        .def_property_readonly("n_basis", &Model::n_basis)
        .def_property_readonly("n_lebedev", &Model::n_lebedev)
        .def_property_readonly("lmax", &Model::lmax)
        .def_property_readonly("cavity", &Model::cavity)
        .def_property_readonly("solvent_epsilon", &Model::solvent_epsilon)
        .def_property_readonly("model", &Model::model)
        .def("scaled_ylm",
             py::overload_cast<array_f_t, int>(&Model::scaled_ylm, py::const_), "coord"_a,
             "sphere"_a, "TODO Docstring")
        .def("scaled_ylm",
             py::overload_cast<array_f_t, int, py::array_t<double>>(&Model::scaled_ylm,
                                                                    py::const_),
             "coord"_a, "sphere"_a, "out"_a, "TODO Docstring")
        .def("solute_nuclear_contribution", &Model::solute_nuclear_contribution,
             "Return the terms of the nuclear contribution to the solvation as a python "
             "dictionary.")
        .def("compute", &Model::compute, "phi"_a, "psi"_a, "tol"_a = 1e-10,
             "maxiter"_a = 100, "ndiis"_a = 20, "print"_a = 0);
}

PYBIND11_MODULE(pyddx, m) {
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
  export_pyddx_classes(m);
  export_pyddx_data(m);
}
